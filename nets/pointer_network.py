import optparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h, batch_size x hidden_dim
        :param Tensor context: Attention context, batch_size x (graph_size+n_depot+n_agent) x hidden_dim
        :param ByteTensor mask: Selection mask, batch_size x 1 x graph_size+n_depot+n_agent
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, (graph_size+n_depot+n_agent))
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, (graph_size+n_depot+n_agent))
        context = context.permute(0, 2, 1)
        # batch_size x hidden_dim x (graph_size+n_depot+n_agent)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, (graph_size+n_depot+n_agent))
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        mask = mask[:, 0, :]
        att[mask] = -math.inf
        alpha = F.softmax(att, dim=-1)
        log_p = F.log_softmax(att, dim=-1)
        # batch_size x hidden_dim
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)
        '''if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping'''
        # logits进行一次mask

        return hidden_state, alpha, log_p

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class PointerNet(nn.Module):

    def __init__(self, 
                 problem,
                 opts):
        super(PointerNet, self).__init__()
        self.model_name = 'pointernet'
        self.problem = problem
        self.opts = opts
        self.shrink_size = opts.shrink_size
        self.encoder = Encoder(opts.embedding_dim,
                                opts.hidden_dim,
                                opts.n_layers,
                                opts.dropout,
                                False)
        self.init_embed_depot = nn.Linear(opts.space_dim, opts.embedding_dim)
        self.init_embed_load = nn.Linear(opts.space_dim, opts.embedding_dim)
        self.init_embed = nn.Linear(opts.space_dim+1, opts.embedding_dim)
        
        self.input_to_hidden = nn.Linear(opts.embedding_dim, 4 * opts.hidden_dim)
        self.hidden_to_hidden = nn.Linear(opts.hidden_dim, 4 * opts.hidden_dim)
        self.hidden_out = nn.Linear(opts.hidden_dim * 2, opts.hidden_dim)
        self.att = Attention(opts.hidden_dim, opts.hidden_dim)

        

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, opts=None, baseline=None, bl_val=None, n_EG=None, return_pi=False, epoch=0, is_baseline=False):
        
        costs, lls = [], []
        
        # input: dict={'loc': tensor.shape = (batch_size, graph_size+n_agent, 2),
        #              'demand': tensor.shape = (batch_size, graph_size+n_agent),
        #              'depot': tensor.shape = (batch_size, n_depot)}
        # TODO: 删除一些冗余的输入参数
        states = self.problem.make_state(input, opts=opts)
        output, sequence = [], []
        agent = []

        # batch_size x (n_depot+graph_size+n_agent) x embedding_dim
        input_embed = self._init_embed(input, self.opts)
        
        # n_layers x batch_size x hidden_dim
        encoder_hidden0 = self.encoder.init_hidden(input_embed)

        # encoder_outputs: batch_size x graph_size+n_depot+n_agent x hidden_dim
        # encoder_hidden: n_layers x batch_size x hidden_dim
        encoder_outputs, encoder_hidden = self.encoder(input_embed, encoder_hidden0)
        
        if opts.bidir:
            decoder_hidden = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                               torch.cat(encoder_hidden[1][-2:], dim=-1))
        else:
            # batch_size x hidden_dim
            decoder_hidden = (encoder_hidden[0][-1].unsqueeze(0),
                               encoder_hidden[1][-1].unsqueeze(0))
        j = 0
        
        # batch_size: 当前时间步是哪个agent, 从agent0开始, 或说是上一时刻的agent
        current_agent = torch.zeros(states.agent_length.shape[0], dtype=torch.int64, device=states.agent_length.device)
        # shrink_size: 在state要结束的时候对batchs_size进行shrink，options里面设置
        # shrink_size不是None或当前path的state还没结束
        # TODO: 变成单机？
        while not (self.shrink_size is None and states.all_finished(opts)):
            # batch_size x 1    (当前agent的已行进路径长)
            current_length = states.agent_length.gather(1, current_agent[:, None])
            # batch_size    (如果当前agent路径长超过了阈值就)
            current_agent = torch.where(current_length.squeeze(1) < opts.mean_distance, current_agent, torch.clamp(current_agent+1, 0, opts.n_agent-1))
            agent.append(current_agent)
            states = states._replace(prev_a=states.agent_prev_a.gather(1, current_agent[:, None]),
                                     used_capacity=states.agent_used_capacity.gather(1, current_agent[:, None])
                                     )
            
            # batch_size x 1 x embed_dim
            current_embed = input_embed.gather(1, states.prev_a[:, None, :].repeat(1, 1, opts.embedding_dim))
            
            # current_embed: batch_size x 1 x embedding_dim
            # decoder_hidden: batch_size x hidden_dim
            # encoder_outputs: batch_size x graph_size+n_depot+n_agent x hidden_dim
            log_p, mask, h_t, c_t = self._get_log_p(current_embed, decoder_hidden, encoder_outputs, states, opts)
            
            decoder_hidden = (h_t, c_t)
            # selected: batch_size，每个batch的graph被选中的点的index
            selected = self._select_node(log_p.exp(), mask[:, 0, :], opts, epoch) 
            
            
            states = states.update(selected, current_agent, opts)
            output.append(log_p)
            sequence.append(selected)
            j += 1
        # batch_size x len(output)=len(sequence) x graph_size+n_depot+n_agent, [i, j]是第i个batch的第j次decode的每个点访问的概率
        _log_p = torch.stack(output, 1)
        # batch_size x len(sequence)=len(output), [i,j]是第i个batch第j次encoder选择的点的index
        pi = torch.stack(sequence, 1)
        agent_all = torch.stack(agent, 1)

        # cost: batch_size
        # mask: None
        cost, mask = self.problem.get_costs(input, pi, agent_all, opts.n_agent, states, opts)

        if return_pi:
            return cost, pi, agent_all

        if baseline != None:
            return cost, _log_p, pi, mask

        return cost, _log_p
        

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        # mask一定是None
        # _log_p = log_p: batch_size x len(output)=len(sequence) x graph_size+n_depot_n_agent
        # a = pi: batch_size x len(sequence)=len(output)
        # a.unsqueeze(-1): batch_size x len(sequence)=len(output) x 1
        # batch_size x len(sequence): 每个batch的每个被选择的点的对应的概率的对数
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        # mask目前天然=None
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        # 每个batch的概率的对数的和，后面可以直接 通过exp算得概率的似然（概率的乘积）
        # batch_size
        return log_p.sum(1)

    def _init_embed(self, input, opts):
        """
        图信息经过线性层
        Args:
            input (dict): loc+depot+demand

        Returns:
            tensor: 图信息的初始embed
        """
        features = ('demand', )
        return torch.cat(
            (
                self.init_embed_depot(input['depot']),  # batch_size x n_depot x embed_dim
                # TODO: initial embedding中加入load charger的embed
                self.init_embed_load(input['loc'][:, :opts.n_agent]),
                self.init_embed(torch.cat((
                    input['loc'][:, opts.n_agent:],
                    *(input[feat][:, opts.n_agent:, None] for feat in features)
                ), -1))   #  batch_size x graph_size+n_agent x embed_dim
            ),
            1
        )



    def _select_node(self, probs, mask, opts, epoch=0):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            # 但这个应该不会取到被mask的元素，因为log_p（即probs）本身是经过mask的，被mask的元素极小不可能被取到
            # _: 每个batch的最大值，batch_size
            # selected: 每个batch对应最大值的index，batch_size
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        
        elif self.decode_type == "sampling":
            # batch_size
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            # 如果sample到的是需要被mask掉的元素就重新sample一次
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected



    def _get_log_p(self, x, hidden, context, state, opts):
        """
        x: batch_size x 1 x embed_dim
        hidden: cell and hidden, m_layers x batch_size x hidden_dim
        context: batch_size x n_depot+graph_size+n_agent x hidden_dim
        """
        # batch_size x hidden_dim
        h, c = hidden
        h, c = h.squeeze(0), c.squeeze(0)

        # batch_size x embed_dim
        x = x.squeeze(1)
        
        # batch_size x 4*hidden_dim
        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        # batch_size x hidden_dim
        input, forget, cell, out = gates.chunk(4, 1)

        input = torch.sigmoid(input)
        forget = torch.sigmoid(forget)
        cell = torch.tanh(cell)
        out = torch.sigmoid(out)

        # batch_size x hidden_dim
        c_t = (forget * c) + (input * cell)
        # batch_size x hidden_dim
        h_t = out * torch.tanh(c_t)


        mask = state.get_mask(opts) 
        # Attention section
        hidden_t, _, log_p = self.att(h_t, context, torch.eq(mask, 1))
        hidden_t = torch.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))
        # mask: batch_size x 1 x graph_size+n_depot+n_agent    (1表示要被mask的，0表示正常的点，已经访问过的和)
        
        assert not torch.isnan(log_p).any()
        # log_p: batch_size x num_steps x graph_size+1
        # mask: batch_size x 1 x graph_size+1
        return log_p, mask, hidden_t, c_t


















