import torch
from torch import nn
import torch.nn.functional as F
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


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 n_agent=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_tsp = problem.NAME == 'tsp'
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.n_agent = n_agent

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 4  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(3, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, opts=None, baseline=None, bl_val=None, return_pi=False):
        """
        model forward propagation
        args:
            input: dict={'loc': tensor.shape = (batch_size, graph_size+n_agent, 3),
                         'demand': tensor.shape = (batch_size, graph_size+n_agent),
                         'depot': tensor.shape = (batch_size, n_depot)}
            opts: parameter configuration
            baseline(class):
        return:
        """
        costs, lls = [], []
        
        # input: dict={'loc': tensor.shape = (batch_size, graph_size+n_agent, 2),
        #              'demand': tensor.shape = (batch_size, graph_size+n_agent),
        #              'depot': tensor.shape = (batch_size, n_depot)}
        states = self.problem.make_state(input, opts=opts)
        output, sequence, agent = [], [], []

        # self._init_embed(input): batch_size x n_depot+graph_size+n_agent x embedding_dim  (图信息的初始embedding)
        # embeddings: batch_size x n_depot+graph_size+n_agent x embed_dim   (图信息的最终embedding)
        # init_context: batch_size x embedding_dim  (embeddings在第二维求平均)
        # attn: n_heads x batch_size x graph_size+n_depot+n_agent x graph_size+n_depot+n_agent    (基于最后一维softmax)
        # v: n_heads x batch_size*(n_depot+n_agent+graph_size) x val_dim    (QKV中的V)
        # h_old: batch_size x n_depot+graph_size+n_agent x embedding_dim    (原始transformer的encoder输出)
        embeddings, init_context, attn, V, h_old = self.embedder(self._init_embed(input))
        
        # fixed = AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
        # embeddings: batch_size x n_depot+graph_size+n_agent x embed_dim   (图信息的最终embedding)
        # fixed context = (batch_size, 1, embed_dim), 对embeddings的第二维求平均后经过一个nn.Linear后再拓展成三维
        # *fixed_attention_node_data: (glimpse_key, glimpse_val, logit_key), 对embeddings的直接拆分以及维数顺序重组
        # logit_key_fixed.contiguous(): batch_size x 1 x graph_size+n_depot+n_agent x embed_dim
        # glimpse_key和glimpse_val: n_heads x batch_size x num_steps x graph_size+1 x val_dim
        fixed = self._precompute(embeddings)
        j = 0

        # batch_size: 当前时间步是哪个agent, 从agent0开始, 或说是上一时刻的agent
        current_agent = torch.zeros(states.agent_length.shape[0], dtype=torch.int64, device=states.agent_length.device)
        # shrink_size: 在state要结束的时候对batchs_size进行shrink，options里面设置
        # shrink_size不是None或当前path的state还没结束
        while not (self.shrink_size is None and states.all_finished(opts)):
            # batch_size x 1    (当前agent的已行进路径长)
            current_length = states.agent_length.gather(1, current_agent[:, None])
            # batch_size    (如果当前agent路径长超过了阈值就)
            current_agent = torch.where(current_length.squeeze(1) < opts.mean_distance,
                                        current_agent, torch.clamp(current_agent+1, 0, opts.n_agent-1))
            agent.append(current_agent)
            states = states._replace(prev_a=states.agent_prev_a.gather(1, current_agent[:, None]),
                                     used_capacity=states.agent_used_capacity.gather(1, current_agent[:, None]))


            # log_p: batch_size x 1 x graph_size+n_depot+n_agent  (每个点被选择的概率)
            # mask: batch_size x 1 x graph_size+n_depot+n_agent
            log_p, mask = self._get_log_p(fixed, states, opts)
            
            # selected: batch_size，每个batch的graph被选中的点的index
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :]) 
            
            states = states.update(selected, current_agent, opts)
            
            output.append(log_p[:, 0, :])
            sequence.append(selected)

            j += 1
        # batch_size x len(output)=len(sequence) x graph_size+n_depot+n_agent
        _log_p = torch.stack(output, 1)
        # batch_size x len(sequence)=len(output)
        pi = torch.stack(sequence, 1)
        agent_all = torch.stack(agent, 1)

        # cost: batch_size
        # mask: None
        cost, mask = self.problem.get_costs(input, pi, agent_all, self.n_agent, states, opts)

        if return_pi:
            return cost, pi, agent_all

        if baseline != None:
            return cost, _log_p, pi, mask


        return cost, _log_p

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

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

    def _init_embed(self, input):
        """
        图信息经过线性层
        Args:
            input (dict): loc+depot+demand

        Returns:
            tensor: 图信息的初始embed
        """
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot']),  # batch_size x n_depot x embed_dim
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))   #  batch_size x graph_size+n_agent x embed_dim
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

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

    def _precompute(self, embeddings, num_steps=1):
        '''
        The fixed context projection of the graph embedding is calculated only once for efficiency
        embeddings: batch_size x n_depot+graph_size+n_agent x embed_dim   (图信息的最终embedding)
        '''

        # TODO: 直接加入一维输入，不用在这边求均值
        # batch_size x embed_dim
        graph_embed = embeddings.mean(1)  

        # batch_size x 1 x embed_dim.   (一个线性层)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # 将embedding映射成(batch_size x 1 x graph_size+n_depot+n_agent x 3*embed_dim)，
        # 然后分成3个(batch_size x 1 x graph_size+n_depot+n_agent x embed_dim)的tensor
        # glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed：(batch_size x 1 x graph_size+n_depot+n_agent x embed_dim)
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        # self._make_heads(glimpse_key_fixed, num_steps): n_heads x batch_size x num_steps x graph_size+1 x val_dim
        # self._make_heads(glimpse_val_fixed, num_steps): n_heads x batch_size x num_steps x graph_size+1 x val_dim
        # logit_key_fixed.contiguous(): batch_size x 1 x graph_size+1 x embed_dim
        # 这里对tensor进行重组时默认val_dim = key_dim
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        # 一种存储数据的方式，改写了namedtuple的__getitem__
        fixed = AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
        return fixed

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, opts, normalize=True):

        # fixed.context_node_projected：batch_size x 1 x embed_dim
        # project_step_context: embed_dim+1 => embed_dim 
        # fixed.node_embeddings: embeddings, batch_size x graph_size+n_depot+n_agent x embed_dim
        # self._get_parallel_step_context(fixed.node_embeddings, state)：batch_size x 1 x embed_dim+1，每个batch
        # 当前所在的点的embed_dim和当前剩余的capacity
        # self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state)):
        # query: batch_size x 1 x embed_dim, 图信息+当前所在点embed和capacity信息
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # 如果不是sdvrp问题的话，就是fixed中的fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key，
        # 对应维数
        # glimpse_K=fixed.glimpse_key: n_heads x batch_size x num_steps x graph_size+n_depot+n_agent x val_dim
        # glimpse_V=fixed.glimpse_val: n_heads x batch_size x num_steps x graph_size+n_depot+n_agent x val_dim
        # logit_K=fixed.logit_key: batch_size x 1 x graph_size+n_depot+n_agent x embed_dim
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # mask: batch_size x 1 x graph_size+n_depot+n_agent    (1表示要被mask的，0表示正常的点，已经访问过的和)
        mask = state.get_mask(opts) 

        # log_p=logits: batch_size x num_steps x graph_size+1
        # glimpse: (batch_size, num_steps, embed_dim)
        # TODO: glimpse没用到
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        # 就是True捏
        # self.temp就是1
        # log_p: batch_size x num_steps x graph_size+1，沿最后一维进行log softmax（在softmax基础和上再做一个log）
        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)
        if torch.isnan(log_p).any():
            for i in range(log_p.shape[0]):
                if torch.isnan(log_p[i]).any():
                    print(i)
                    print(mask[i])
                    print(state.prev_a[i])
                    print(state.used_capacity[i])
                    print(state.coords[i, state.prev_a[i]])
                    print(state.coords[i, :opts.n_depot])
        assert not torch.isnan(log_p).any()
        # log_p: batch_size x num_steps x graph_size+1
        # mask: batch_size x 1 x graph_size+1
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        这个好像是用来根据state中当前的点的index选择出其对应embedding的
        只有在num_steps>1才能称作parallel
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        # 是类里面的prev_a
        # 初始化的是全为0的tensor，batch_size x 1
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            # 代码中就是默认为False中是False
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                # batch_size x 1 x embed_dim+1，就是除了点的embed之外再加上当前货车剩余的capcity

                return torch.cat(
                    (
                        torch.gather(  # cat的第一个tensor，
                            embeddings,
                            1,
                            current_node.contiguous()
                                        .view(batch_size, num_steps, 1)
                                        .expand(batch_size, num_steps, embeddings.size(-1))   # prev_a：转换成 batch_size x 1 x embed_dim
                                    ).view(batch_size, num_steps, embeddings.size(-1)),  # 基于prev_a(也就是current_node)将每个batch中对应的embedding选出来

                            self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]  # cat的第二个tensor
                    ),
                    -1
                )

        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        # num_steps=1，图信息+当前所在点embed和capacity信息
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # n_heads x batch_size x 1 x 1 x val_dim   (query分head)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        
        # glimpse_K=fixed.glimpse_key: n_heads x batch_size x 1 x graph_size+n_depot+n_agent x val_dim
        # glimpse_V=fixed.glimpse_val: n_heads x batch_size x 1 x graph_size+n_depot+n_agent x val_dim
        # logit_K=fixed.logit_key: batch_size x 1 x graph_size+n_depot+n_agent x embed_dim
        # n_heads x batch_size x 1 x 1 x graph_size+n_depot+n_agent
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        
        # TODO: 是否要进行这一步mask？
        # 先是compatibility进行一次mask
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            # 每个head是同样的位置被mask
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # TODO: 查看点数比较少的图中的heads情况
        # n_heads x batch_size x 1 x 1 x val_dim
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        #   (batch_size, 1, 1, n_heads, val_size)
        # =>(batch_size, 1, 1, n_heads*val_size=embed_dim)
        # =>(batch_size, 1, 1, embed_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # 上面的计算得到最终的query
        final_Q = glimpse
        # logits: batch_size x 1 x graph_size+n_depot+n_agent
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # 这个if也是True
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        # logits进行一次mask
        if self.mask_logits:
            logits[mask] = -math.inf
        # logits: batch_size x 1 x graph_size+n_depot+n_agent
        # glimpse.squeeze(-2): (batch_size, 1, embed_dim)
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        # QKV拆成多个head
        # 输入维数是 batch_size x 1 x graph_size+n_depot+n_agent x embed_dim
        # 最终维数：n_heads x batch_size x num_steps x graph_size+n_depot+n_agent x val_dim
        # 这里n_heads*val_dim = embed_dim
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
