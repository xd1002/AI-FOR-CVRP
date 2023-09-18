import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    agent_length: torch.Tensor
    agent_used_capacity: torch.Tensor
    agent_prev_a: torch.Tensor
    current_distance: torch.Tensor
    mask_depot: torch.Tensor
    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, n_agent, opts, visited_dtype=torch.uint8):
        # input: dict={'loc': tensor.shape = (batch_size, graph_size+n_agent, 3),
        #              'demand': tensor.shape = (batch_size, graph_size+n_agent),
        #              'depot': tensor.shape = (batch_size, n_depot)}
        depot = input['depot']
        loc = input['loc']
        demand = input['demand']
        batch_size, n_loc, _ = loc.size()
        # batch_size x 1 x graph_size+n_agent+n_depot
        visited_ = torch.zeros(batch_size, 1, n_loc + opts.n_depot, dtype=torch.uint8, device=loc.device)
        # 开始点mask掉
        visited_[:, :, opts.n_depot:opts.n_depot+opts.n_agent] = 1
        return StateCVRP(
            coords=torch.cat((depot, loc), -2),  # batch_size x graph_size+n_depot+n_agent x 3  (坐标，起点+depot+访问点)
            demand=demand,  # batch_size x graph_size+n_agent   (demand，起点+访问点)
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),     # 各batch当前agent的所在点
            used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  
                visited_
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['loc'][:, opts.n_depot:opts.n_depot+opts.n_agent, :],  
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  
            agent_length=torch.zeros((batch_size, n_agent), device=loc.device),
            agent_used_capacity=demand.new_zeros(batch_size, n_agent),
            agent_prev_a=torch.arange(n_agent, dtype=torch.int64, device=loc.device).view(1, -1).repeat(batch_size, 1) + opts.n_depot,  # 每个agent的当前所在点
            current_distance=loc.new_zeros(demand.shape),
            mask_depot=torch.zeros(batch_size, depot.shape[1], dtype=torch.bool, device=loc.device)
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, current_agent, opts):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # batch_size x 1
        selected = selected[:, None]  
        prev_a = selected
        # graph_size+n_agent
        n_loc = self.demand.size(-1)  

        agent_prev_a = self.agent_prev_a.scatter(-1, current_agent.view(-1, 1), prev_a)
        # batch_size x graph_size+n_depot+n_agent x 2
        cur_coord = self.coords[self.ids, selected]
        # self.lengths: batch_size x 1, 存放当前每个batch的路径长度
        # batch_size x 1
        current_agent_length = self.agent_length.gather(1, current_agent.view(-1, 1))
        # batch_size x 1
        selected_prev = self.agent_prev_a.gather(1, current_agent.view(-1, 1))
        # batch_size x 1 x 3
        prev_coord = self.coords[self.ids, selected_prev]
        selected_distance = (cur_coord - prev_coord).norm(p=2, dim=-1)
        # 如果所有点都遍历完了就不计距离
        lengths = current_agent_length + selected_distance * (1 - self.visited_[:, :, opts.n_depot:].all(axis=-1))
        agent_length = self.agent_length.scatter(-1, current_agent.view(-1, 1), lengths)


        
        # selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        # 这个torch.clamp(prev_a - 1, 0, n_loc - 1)是干啥的
        current_used_capacity = self.agent_used_capacity.gather(1, current_agent.view(-1, 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - opts.n_depot, 0, n_loc - 1)]
        used_capacity = (current_used_capacity + selected_demand + selected_distance * opts.dist_coef)
        for i in range(opts.n_depot):
            used_capacity = used_capacity * (prev_a != i).float()
        agent_used_capacity = self.agent_used_capacity.scatter(-1, current_agent.view(-1, 1), used_capacity)

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            # prev_a[:, :, None]: batch_size x 1 x 1
            # 按照prev_a的元素来，prev_a中[x, z, y]位置处的元素为k，则将1填入self.visited_的[x, y, k]处
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            agent_prev_a=agent_prev_a, agent_used_capacity=agent_used_capacity, visited_=visited_,
            agent_length=agent_length, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self, opts):
        # self.i是啥
        # demand: batch_size x graph_size
        # self.visited是个property function，输出的是那个self.visited_，要求全部被mask为1
        return self.i.item() >= self.demand.size(-1) and self.visited[:, :, opts.n_depot:].all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self, opts):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # self.visited_: batch_size x 1 x graph_size + n_agent  (depot外的其他点的mask情况)
        # 默认的是第一个，else就不用管了
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, opts.n_depot:] 
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        # TODO: 是当前所在点吗
        # batch_size x 1    (当前所在点)
        current_node = self.get_current_node()
        # batch_size x 1 x 3    (当前所在点的坐标)
        current_node_coords = self.coords.gather(1, current_node[:, :, None].repeat(1, 1, 3))
        # batch_size x graph_size+n_agent   (当前点到除了depot外其他所有点到距离)
        self.current_distance[:] = (self.coords[:, opts.n_depot:, :] - current_node_coords).norm(p=2, dim=-1)
        # batch_size x n_depot x graph_size+n_agent   (当前点到除了depot外其他所有点到距离+所有点到各个depot到距离)
        # 防止下一步的电量不支持无人机去向任意一个点(包括充电桩)
        future_distance = self.current_distance.unsqueeze(1).repeat(1, opts.n_depot, 1) + \
                         (self.coords[:, opts.n_depot:, :].unsqueeze(1).repeat(1, opts.n_depot, 1, 1) -
                          self.coords[:, :opts.n_depot, :].unsqueeze(2)).norm(p=2, dim=-1)
        # batch_size x n_depot x graph_size+n_agent   (当前demand包含下一个点的demand+到下一个点和下一个点到depot的总损失电量, 即考虑未来信息)
        # TODO: demand更新的时候有没有将future_distance算入
        current_demand = self.demand.unsqueeze(1).repeat(1, opts.n_depot, 1) + future_distance * opts.dist_coef
        # batch_size x 1 x graph_size+n_agent   (mask掉之前已经访问的和当前到不了的)
        mask_loc = (
            visited_loc |
            (torch.sum((current_demand + self.used_capacity.unsqueeze(1).repeat(1, opts.n_depot, 1) > self.VEHICLE_CAPACITY - opts.safe_coef), axis=1) == opts.n_depot)[:, None, :]
        )

        # batch_size x n_depot  (当前所在节点（或称作上一个节点）就是depot或者上面的mask_loc每行中还有元素为0（未被访问）就mask掉depot)
        for i in range(opts.n_depot):
            self.mask_depot[:, i] = (self.prev_a.squeeze(1) == i) & ((mask_loc == 0).int().sum(-1) > 0).squeeze(1)

        # batch_size x n_depot  (当前所在点到各个depot的距离)
        dist_to_depot = (self.coords[:, :opts.n_depot, :] - current_node_coords).norm(p=2, dim=-1)
        # batch_size x n_depot  (当前所在点到各个depot所需capacity)
        to_depot_used_capacity = self.used_capacity.repeat(1, opts.n_depot) + dist_to_depot * opts.dist_coef
        # 一方面mask掉上次在depot且还有可以访问的点的情况，只要有一个depot因为这种情况被mask则其余depot也要被mask
        # 在此基础上，mask掉能耗不够到达的depot（上面mask_loc中保证每一步一定能到达其中一个depot）
        self.mask_depot[:] = ((~torch.prod(~self.mask_depot, axis=1, dtype=bool).view(-1, 1).repeat(1, opts.n_depot)) | (to_depot_used_capacity > self.VEHICLE_CAPACITY))

        # 这部分只是test使用
        final_mask = torch.cat((self.mask_depot[:, None, :], mask_loc > 0),-1)
        if ((final_mask == True).all(axis=-1) == True).any():
            for i in range(final_mask.shape[0]):
                if (final_mask[i] == True).all():
                    print(to_depot_used_capacity[i])

        # self.mask_depot = torch.ones_like(self.mask_depot)
        return torch.cat((self.mask_depot[:, None, :], mask_loc > 0), -1)

    def construct_solutions(self, actions):
        return actions
