from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, agent_all, n_agent, state, opts):
        '''
        验证除了depot外的每个点被访问且仅被访问一次，验证在结束前车没有超过capacity，
        '''
        # dataset: 那个dict类型的input
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        # 每个batch每步选点的index
        # raw data: [1, 0, 2, 1, 0, 2]
        # value: [0, 0, 1, 1, 2, 2]
        # index: [1, 4, 0, 3, 2, 5]
        # 输出是每个batch排序后的value（不是index）
        # batch_size x len(sequence)
        sorted_pi = pi.data.sort(1)[0]
        # Sorting it should give all zeros at front and then 1...n
        # 要求pi中1~20出现且仅出现一次（上面pi在sort完后与一个标准数组进行比较，正确情况应该是pi的[i, -graph_size:]为1,...,20），
        # 同时其余的都是0(pi在sort后[i, :-graph_size])全是0
        assert (
            torch.arange(opts.n_depot+opts.n_agent, graph_size + opts.n_depot, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size-opts.n_agent) ==
            sorted_pi[:, -graph_size+opts.n_agent:]
        ).all() and (sorted_pi[:, :-graph_size+opts.n_agent] < opts.n_depot).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        # axis=1:其余index分量不变，变第二维的分量，如a.shape=(10,2),b.shape=(10,3),torch.cat((a,b),axis=1)开一个新tensor，
        # [:,0:2]放a,[:,0+2:3+2]放b
        # batch_size x graph_size+1 , 在原来的input['demand']后面加上有关depot的demand(新的tensor)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :opts.n_depot], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        # batch_size x len(sequence)
        # d中元素[i, j]表示第i个batch对应的点pi[i, j]的demand d[i, j]
        # 这个不是前面sort过的而是打乱顺序的
        d = demand_with_depot.gather(1, pi)
        # batch_size
        used_cap = torch.zeros(pi.shape[0], n_agent).cuda()
        # 看每一步的demand是否满足要求
        for i in range(pi.size(1)):
            cur_cap = used_cap.gather(1, agent_all[:, i].view(-1, 1))
            cur_cap += d[:, i][:, None]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap = used_cap.scatter(1, agent_all[:, i].view(-1, 1), cur_cap)
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        # batch_size x graph_size+1 x 2：包括depot在内每个点的坐标（第一个是depot，然后是其余目标点）
        #loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        # 每个batch的每个step的点的坐标选出来
        # batch_size x len(sequence) x 2
        #d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        # batch_size, 每个batch_size的reward：路径长度（后面两个加法项：第一个是因为pi的index是从离开depot的第一个点开始的，
        # 相当于少加了第一段路的长度；第二个：本来pi的第二维就由）
        return state.agent_length.sum(axis=-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # 检验是否所有点都被访问
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, opts=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        # 如果是用别人提供的数据
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
        # 如果在程序中生成数据
        else:

            # From VRP with RL paper https://arxiv.omake_datasetrg/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 3000.,
                50: 40.,
                100: 50.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size+opts.n_agent, 3).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size+opts.n_agent).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(opts.n_depot, 3).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
