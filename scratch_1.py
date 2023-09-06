##
import torch
from torch import nn
from typing import NamedTuple
import numpy as np
import math
import os
import pandas as pd
import copy
from matplotlib import pyplot as plt
from typing import NamedTuple
##
embed_dim=128
graph_size=20
input_dim=128
batch_size=512
n_heads=2
n_query=graph_size
node_dim=3
features = ('demand', )
n_layers=10
normalization='batch'
val_dim=int(embed_dim/n_heads)
num_steps=1

##
input = {}
input['loc'] = torch.rand((batch_size, graph_size, 2))
input['demand'] = torch.rand((batch_size, graph_size))
input['depot'] = torch.rand((batch_size, 2))
##
init_embed = nn.Linear(node_dim, embed_dim)
a = init_embed(torch.cat((input['loc'], *(input[feat][:, :, None] for feat in features)), -1).shape).shape
##
embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            normalization=normalization
        )

data=torch.rand((batch_size, graph_size+1, embed_dim))
process=embedder(data)
print(process[0].shape, process[1].shape, process[2].shape, process[3].shape, process[4].shape)
##
mha=MultiHeadAttention(n_heads, input_dim, embed_dim=embed_dim)
mha(data).shape
##
embeddings=torch.rand((batch_size , graph_size+1 , embed_dim))
graph_embed = embeddings.mean(1)
print(graph_embed.shape)
##
b=torch.rand((10,2))
a={1:b}
d=torch.rand((10,2))
c={1:d}
torch.equal(a[1],c[1])
##
ids = torch.arange(batch_size)[:, None]
selected=torch.randint(0,2,size=(batch_size, 1))
coords=torch.rand((batch_size, graph_size+1, 2))
cur_coord = coords[ids, selected].shape
cur_coord
##
b = torch.zeros((batch_size, graph_size+10))
for i in range(batch_size):
    b[i, :] = torch.randint(0, 20, size=(1,30))
print(b[0])
print(b.sort(1))
##
pi=torch.rand((batch_size, 36))
loc_with_depot=torch.rand((batch_size, graph_size+1, 2))
pi[...,None].expand(*pi.size(),loc_with_depot.size(-1)).shape

##
a=np.random.randint(0,10,(2,10))
path=os.path.join('tests','cvrp_20')
np.savetxt('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\'
           'MDAM-master\\MDAM-master\\tests\\cvrp_20\\test_data.csv', a, delimiter=',')
##
def fun():
    return 1, 2
fun()[0]
##
# 只画一幅图
os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master\\tests\\cvrp_21\\run_20221026T180911')
data21 = pd.read_csv('routine.csv', header=None)
graph21 = pd.read_csv('graph_info.csv', header=None)
d21 = np.array(data21)
g21 = np.array(graph21)
target_d21 = d21[0, :]
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.plot(g21[[0, int(target_d21[0])], 0], g21[[0, int(target_d21[0])], 1], color='red')
for i in range(len(target_d21)):
    point = int(target_d21[i])
    if i < len(target_d21)-1:
        point_next = int(target_d21[i+1])
        if point == 0:
            if point_next == 0:
                break
            else:
                ax.plot(g21[[point, point_next], 0], g21[[point, point_next], 1], color='red')
        else:
            ax.plot(g21[[point, point_next], 0], g21[[point, point_next], 1], color='red')
    else:
        if point == 0:
            break
        else:
            ax.plot(g21[[point, 0], 0], g21[[point, 0], 1])
plt.title('21 points')
##



os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master\\tests\\cvrp_20\\run_20221026T150231')
data20 = pd.read_csv('routine.csv', header=None)
graph20 = pd.read_csv('graph_info.csv', header=None)
d20 = np.array(data20)
g20 = np.array(graph20)
target_d20 = d20[0, :]
ax2 = fig.add_subplot(122)
ax2.plot(g20[[0, int(target_d20[0])], 0], g20[[0, int(target_d20[0])], 1], color='red')
for i in range(len(target_d20)):
    point = int(target_d20[i])
    if i < len(target_d20)-1:
        point_next = int(target_d20[i+1])
        if point == 0:
            if point_next == 0:
                break
            else:
                ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
        else:
            ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
    else:
        if point == 0:
            break
        else:
            ax2.plot(g20[[point, 0], 0], g20[[point, 0], 1])
plt.title('20 points')
# ax.quiver(0,0,1,1,color=(1, 0, 0, 0.3),angles='xy', scale_units='xy', scale=1)
# plt.xlim((-2,2))
# plt.ylim((-2,2))
os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master\\image')
plt.savefig('./21点和20点同一个模型下比较.jpg', dpi=300)
plt.show(block=True)


##
hh = 3
ww = 3
fig, axes = plt.subplots(hh, ww, figsize=(10, 10))
axes_list = []
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes_list.append(axes[i, j])
os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\'
         'MDAM-master\\MDAM-master\\tests\\cvrp_21\\run_20221026T195455')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
data = np.array(data)
graph = np.array(graph)
for idx, ax2 in enumerate(axes_list):
    g20 = graph[22*idx:22*(idx+1)]
    target_d20 = data[idx, :]
    ax2.plot(g20[[0, int(target_d20[0])], 0], g20[[0, int(target_d20[0])], 1], color='red')
    for i in range(len(target_d20)):
        point = int(target_d20[i])
        if i < len(target_d20) - 1:
            point_next = int(target_d20[i + 1])
            if point == 0:
                if point_next == 0:
                    break
                else:
                    ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
            else:
                ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
        else:
            if point == 0:
                break
            else:
                ax2.plot(g20[[point, 0], 0], g20[[point, 0], 1])
    ax2.set_title('({:.4f}, {:.4f})'.format(g20[-1, 0], g20[-1, 1]))
    ax2.scatter(g20[-1, 0], g20[-1, 1], color='black', linewidth=3)

# 第一幅图差距最小，将第一幅图转化为原图
os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master\\tests\\cvrp_20\\run_20221026T150231')
data20 = pd.read_csv('routine.csv', header=None)
graph20 = pd.read_csv('graph_info.csv', header=None)
d20 = np.array(data20)
g20 = np.array(graph20)
target_d20 = d20[0, :]
ax2 = axes_list[0]
ax2.clear()
ax2.plot(g20[[0, int(target_d20[0])], 0], g20[[0, int(target_d20[0])], 1], color='red')
for i in range(len(target_d20)):
    point = int(target_d20[i])
    if i < len(target_d20)-1:
        point_next = int(target_d20[i+1])
        if point == 0:
            if point_next == 0:
                break
            else:
                ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
        else:
            ax2.plot(g20[[point, point_next], 0], g20[[point, point_next], 1], color='red')
    else:
        if point == 0:
            break
        else:
            ax2.plot(g20[[point, 0], 0], g20[[point, 0], 1])
ax2.set_title('original graph: 20 points')

os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master\\image')
plt.savefig('./21点和20点同一个模型下比较(更多采样).jpg', dpi=300)
plt.show(block=True)
##
# 3d作图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master-to-multi-car\\tests\\cvrp_20\\run_20221106T205915')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
data = np.array(data)
graph = np.array(graph)

idx = 0
g20 = graph[21*idx:21*(idx+1)]
target_d20 = data[idx, :]
ax.plot3D(g20[[0, int(target_d20[0])], 0], g20[[0, int(target_d20[0])], 1], g20[[0, int(target_d20[0])], 2], color='red')
for i in range(len(target_d20)):
    point = int(target_d20[i])
    if i < len(target_d20) - 1:
        point_next = int(target_d20[i + 1])
        if point == 0:
            if point_next == 0:
                break
            else:
                ax.plot3D(g20[[point, point_next], 0], g20[[point, point_next], 1], g20[[point, point_next], 2], color='red')
        else:
            ax.plot3D(g20[[point, point_next], 0], g20[[point, point_next], 1], g20[[point, point_next], 2], color='red')
    else:
        if point == 0:
            break
        else:
            ax.plot3D(g20[[point, 0], 0], g20[[point, 0], 1], g20[[point, 0], 2])
# ax.set_title('({:.4f}, {:.4f})'.format(g20[-1, 0], g20[-1, 1]))
# ax.scatter(g20[-1, 0], g20[-1, 1], color='black', linewidth=3)
ax.scatter3D(g20[0, 0], g20[0, 1], g20[0, 2], linewidths=5,color='black')
plt.show(block=True)

##
# 3d作图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
n_agent = 3
# run_20221110T195028
# 多机场景下训练
os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-car\\tests\\cvrp_30\\run_20221117T224429')
# 单机场景下训练
# os.chdir('D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master-to-multi-car\\tests\\cvrp_30\\run_20221110T223956')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
agent = pd.read_csv('agent_all.csv', header=None)
data = np.array(data)
graph = np.array(graph)
agent = np.array(agent)
colors = ['red', 'green', 'blue']


length_list_all = []
for idx in [i for i in range(10)]:
    g20 = graph[31*idx:31*(idx+1)]
    prev_a = [0] * n_agent
    trajectory_end = [0] * n_agent
    agent20 = agent[idx, :]
    target_d20 = data[idx, :]
    length = 0
    length_list = [0] * n_agent
    trajectory = [[] for i in range(n_agent)]

    for i in range(len(target_d20)):
        agent_index = int(agent20[i])
        trajectory[agent_index].append(target_d20[i])

    for idx, tra in enumerate(trajectory):
        prev_point = 0
        for idx_tra, now_point in enumerate(tra):
            now_point = int(now_point)
            if now_point == 0:
                if idx_tra == len(tra) - 1:
                    break
                elif tra[idx_tra + 1] == 0:
                    break
                else:
                    length_list[idx] += np.sqrt(np.sum((g20[now_point]-g20[prev_point])**2))
            else:
                length_list[idx] += np.sqrt(np.sum((g20[now_point] - g20[prev_point]) ** 2))
            prev_point = copy.deepcopy(now_point)
    length_list_all.append(sum(length_list))
print('{} +- {}'.format(np.mean(length_list_all), np.std(length_list_all)))

idx = 0
prev_a = [0] * n_agent
g20 = graph[31 * idx:31 * (idx + 1)]
agent20 = agent[idx, :]
target_d20 = data[idx, :]
trajectory = [[] for i in range(n_agent)]

for i in range(len(target_d20)):
    agent_index = int(agent20[i])
    trajectory[agent_index].append(target_d20[i])

for col, tra in enumerate(trajectory):
    prev_point = 0
    for idx, now_point in enumerate(tra):
        now_point = int(now_point)
        if now_point == 0:
            if idx == len(tra)-1:
                break
            elif tra[idx+1] == 0:
                break
            else:
                ax.plot3D(g20[[prev_point, now_point], 0],
                          g20[[prev_point, now_point], 1],
                          g20[[prev_point, now_point], 2],
                          color=colors[col])
        else:
            ax.plot3D(g20[[prev_point, now_point], 0],
                      g20[[prev_point, now_point], 1],
                      g20[[prev_point, now_point], 2],
                      color=colors[col])
        prev_point = copy.deepcopy(now_point)
ax.scatter3D(g20[0, 0], g20[0, 1], g20[0, 2], linewidths=5, color='black')
plt.show(block=True)
# 10.030752549005587 +- 0.5023076580036876
##
# 单机场景单机训练
# 3d作图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
n_agent = 1
os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-3d-0\\tests\\cvrp_30\\run_20221118T181405')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
data = np.array(data)
graph = np.array(graph)
agent = np.zeros((data.shape[0], data.shape[1]))
colors = ['red', 'green', 'blue']


length_list_all = []
for idx in [i for i in range(10)]:
    g20 = graph[31*idx:31*(idx+1)]
    prev_a = [0] * n_agent
    trajectory_end = [0] * n_agent
    agent20 = agent[idx, :]
    target_d20 = data[idx, :]
    length = 0
    length_list = [0] * n_agent
    trajectory = [[] for i in range(n_agent)]

    for i in range(len(target_d20)):
        agent_index = int(agent20[i])
        trajectory[agent_index].append(target_d20[i])

    for idx, tra in enumerate(trajectory):
        prev_point = 0
        for idx_tra, now_point in enumerate(tra):
            now_point = int(now_point)
            if now_point == 0:
                if idx_tra == len(tra) - 1:
                    break
                elif tra[idx_tra + 1] == 0:
                    break
                else:
                    length_list[idx] += np.sqrt(np.sum((g20[now_point]-g20[prev_point])**2))
            else:
                length_list[idx] += np.sqrt(np.sum((g20[now_point] - g20[prev_point]) ** 2))
            prev_point = copy.deepcopy(now_point)
    length_list_all.append(sum(length_list))
print('{} +- {}'.format(np.mean(length_list_all), np.std(length_list_all)))

idx = 0

prev_a = [0] * n_agent
g20 = graph[31 * idx:31 * (idx + 1)]
agent20 = agent[idx, :]
target_d20 = data[idx, :]
trajectory = [[] for i in range(n_agent)]

for i in range(len(target_d20)):
    agent_index = int(agent20[i])
    trajectory[agent_index].append(target_d20[i])

for col, tra in enumerate(trajectory):
    prev_point = 0
    for idx, now_point in enumerate(tra):
        now_point = int(now_point)
        if now_point == 0:
            if idx == len(tra)-1:
                break
            elif tra[idx+1] == 0:
                break
            else:
                ax.plot3D(g20[[prev_point, now_point], 0],
                          g20[[prev_point, now_point], 1],
                          g20[[prev_point, now_point], 2],
                          color=colors[col])
        else:
            ax.plot3D(g20[[prev_point, now_point], 0],
                      g20[[prev_point, now_point], 1],
                      g20[[prev_point, now_point], 2],
                      color=colors[col])
        prev_point = copy.deepcopy(now_point)
ax.scatter3D(g20[0, 0], g20[0, 1], g20[0, 2], linewidths=5, color='black')
plt.show(block=True)
##
# unbalance
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
n_agent = 4
n_depot = 16
# run_20221110T195028
# 多机场景下训练
# os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\tests\\cvrp_30\\多机多起点多depot无balance')
os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\video\\run_20221215T130054')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
agent = pd.read_csv('agent_all.csv', header=None)
data = np.array(data)
graph = np.array(graph)
agent = np.array(agent)
print(agent.shape)
colors = ['red', 'green', 'blue', 'purple']
graph_size = 285
cal_num = 1

length_list_all0 = []
square_error0=[]
for idx in [i for i in range(cal_num)]:
    g20 = graph[(graph_size + n_depot+n_agent)*idx:(graph_size + n_depot+n_agent)*(idx+1)]
    prev_a = [i for i in range(n_depot, n_depot+n_agent)]
    trajectory_end = [0] * n_agent
    agent20 = agent[idx, :]
    target_d20 = data[idx, :]
    length = 0
    length_list = [0] * n_agent
    trajectory = [[] for i in range(n_agent)]

    for i in range(len(target_d20)):
        agent_index = int(agent20[i])
        trajectory[agent_index].append(target_d20[i])
    for idx, tra in enumerate(trajectory):
        prev_point = prev_a[idx]
        for idx_tra, now_point in enumerate(tra):
            now_point = int(now_point)
            if now_point == 0 or now_point == 1 or now_point == 2:
                if idx_tra == len(tra) - 1:
                    break
                elif tra[idx_tra + 1] == 0 or tra[idx_tra + 1] == 1 or tra[idx_tra + 1] == 2:
                    break
                else:
                    length_list[idx] += np.sqrt(np.sum((g20[now_point, :3] - g20[prev_point, :3]) ** 2))
            else:
                length_list[idx] += np.sqrt(np.sum((g20[now_point, :3] - g20[prev_point, :3]) ** 2))
            prev_point = copy.deepcopy(now_point)
    length_list_all0.append(sum(length_list))
    s_mean = np.mean(g20[n_depot:(n_depot+n_agent), :3], axis=0)
    square_error0.append(np.sum((g20[n_depot:(n_depot+n_agent), :3]-s_mean)**2))
print('{} +- {}'.format(np.mean(length_list_all0), np.std(length_list_all0)/math.sqrt(cal_num)))

idx = 0
prev_a = [i for i in range(n_depot, n_depot+n_agent)]
g20 = graph[(graph_size + n_depot+n_agent)*idx:(graph_size + n_depot+n_agent)*(idx+1)]
agent20 = agent[idx, :]
target_d20 = data[idx, :]
trajectory = [[] for i in range(n_agent)]
for i in range(len(target_d20)):
    agent_index = int(agent20[i])
    trajectory[agent_index].append(target_d20[i])

for col, tra in enumerate(trajectory):
    prev_point = prev_a[col]
    for idx, now_point in enumerate(tra):
        now_point = int(now_point)
        if now_point == 0 or now_point == 1 or now_point == 2:
            if idx == len(tra)-1:
                break
            elif tra[idx+1] == 0 or tra[idx+1] == 1 or tra[idx+1] == 2:
                break
            else:
                ax.plot3D(g20[[prev_point, now_point], 0],
                          g20[[prev_point, now_point], 1],
                          g20[[prev_point, now_point], 2],
                          color=colors[col])
        else:
            ax.plot3D(g20[[prev_point, now_point], 0],
                      g20[[prev_point, now_point], 1],
                      g20[[prev_point, now_point], 2],
                      color=colors[col])
        prev_point = copy.deepcopy(now_point)
for i in range(n_depot):
    ax.scatter3D(g20[i, 0], g20[i, 1], g20[i, 2], linewidths=5, color='grey')
for i in prev_a:
    ax.scatter3D(g20[i, 0], g20[i, 1], g20[i, 2], linewidths=5, color='black')
plt.show(block=True)

##balance
#
fig = plt.figure()
pathlist=['E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\result\\285-init-worst',
          'E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\result\\285_tour']
draw_idx = 3
for kk in range(2):
    ax = fig.add_subplot(1, 2, kk+1, projection='3d')
    n_agent = 4
    n_depot = 16
    # run_20221110T195028
    # 多机场景下训练
    #os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\tests\\cvrp_30\\多机多起点多depot无balance')
    #os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\result\\285-init-worst')
    os.chdir(pathlist[kk])
    data = pd.read_csv('routine.csv', header=None)
    graph = pd.read_csv('graph_info.csv', header=None)
    agent = pd.read_csv('agent_all.csv', header=None)
    data = np.array(data)
    graph = np.array(graph)
    agent = np.array(agent)
    colors = ['red', 'green', 'blue', 'purple']
    #colors = ['purple']
    graph_size = 285
    n_depot = 16
    cal_num = 1

    length_list_all1 = []
    square_error1=[]
    for idx in [i for i in range(cal_num)]:
        g20 = graph[(graph_size + n_depot+n_agent)*idx:(graph_size + n_depot+n_agent)*(idx+1)]
        prev_a = [i for i in range(n_depot,n_depot+n_agent)]
        trajectory_end = [0] * n_agent
        agent20 = agent[idx, :]
        target_d20 = data[idx, :]
        length = 0
        length_list = [0] * n_agent
        trajectory = [[] for i in range(n_agent)]

        for i in range(len(target_d20)):
            agent_index = int(agent20[i])
            trajectory[agent_index].append(target_d20[i])
        for idx, tra in enumerate(trajectory):
            prev_point = prev_a[idx]
            for idx_tra, now_point in enumerate(tra):
                now_point = int(now_point)
                if now_point == 0 or now_point == 1 or now_point == 2:
                    if idx_tra == len(tra) - 1:
                        break
                    elif tra[idx_tra + 1] == 0 or tra[idx_tra + 1] == 1 or tra[idx_tra + 1] == 2:
                        break
                    else:
                        length_list[idx] += np.sqrt(np.sum((g20[now_point, :3] - g20[prev_point, :3]) ** 2))
                else:
                    length_list[idx] += np.sqrt(np.sum((g20[now_point, :3] - g20[prev_point, :3]) ** 2))
                prev_point = copy.deepcopy(now_point)
        length_list_all1.append(sum(length_list))
        s_mean=np.mean(g20[n_depot:(n_depot+n_agent), :3],axis=0)
        square_error1.append(np.sum((g20[n_depot:(n_depot+n_agent), :3]-s_mean)**2))
    print('{} +- {}'.format(np.mean(length_list_all1), np.std(length_list_all1)/math.sqrt(cal_num)))
    #print(length_list[draw_idx])

    idx = 0
    prev_a = [i for i in range(n_depot,n_depot+n_agent)]
    g20 = graph[(graph_size + n_depot+n_agent)*idx:(graph_size + n_depot+n_agent)*(idx+1)]
    agent20 = agent[idx, :]
    target_d20 = data[idx, :]
    trajectory = [[] for i in range(n_agent)]
    for i in range(len(target_d20)):
        agent_index = int(agent20[i])
        trajectory[agent_index].append(target_d20[i])
    print(len(trajectory[draw_idx]))
    #trajectory=[trajectory[draw_idx]]
    for col, tra in enumerate(trajectory):
        prev_point = prev_a[col]
        for idx, now_point in enumerate(tra):
            now_point = int(now_point)
            if now_point == 0 or now_point == 1 or now_point == 2:
                if idx == len(tra)-1:
                    break
                elif tra[idx+1] == 0 or tra[idx+1] == 1 or tra[idx+1] == 2:
                    break
                else:
                    ax.plot3D(g20[[prev_point, now_point], 0],
                              g20[[prev_point, now_point], 1],
                              g20[[prev_point, now_point], 2],
                              color=colors[col])
            else:
                ax.plot3D(g20[[prev_point, now_point], 0],
                          g20[[prev_point, now_point], 1],
                          g20[[prev_point, now_point], 2],
                          color=colors[col])
            prev_point = copy.deepcopy(now_point)
    for i in range(n_depot):
        ax.scatter3D(g20[i, 0], g20[i, 1], g20[i, 2], linewidths=5, color='grey')
    for i in prev_a:
        ax.scatter3D(g20[i, 0], g20[i, 1], g20[i, 2], linewidths=5, color='black')
plt.show(block=True)
##
os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\valid')
uav = locals()
for i in range(1, 5):
    uav['uav{}'.format(i)] = np.array(pd.read_csv('uavs_{}.csv'.format(i), header=None))
# uav1: num_depot:2;  length:4.4512879390022615;  depot: [0.38882  0.35775  0.077519] [0.38434  0.56684  0.091606]
# uav2: num_depot:2;  length:4.5284758438173105;  depot: [0.57056  0.78299  0.078128] [0.38434  0.56684  0.091606]
# uav3: num_depot:2;  length:4.3260111550394456;  depot: [0.56976  0.13686  0.049978] [0.74826  0.14358  0.030382]
# uav4: num_depot:1;  length:2.569389328015006;  depot: [0.57056  0.78299  0.078128]


depot = [np.array([0.20481, 0.57646, 0.058227]),
         np.array([0.38434, 0.56684, 0.091606]),
         np.array([0.38909, 0.78342, 0.055505]),
         np.array([0.21057, 0.78864, 0.016025]),
         np.array([0.21031, 0.35338, 0.055076]),
         np.array([0.37974, 0.14122, 0.068012]),
         np.array([0.20141, 0.14121, 0.017383]),
         np.array([0.38882, 0.35775, 0.077519]),
         np.array([0.57056, 0.78299, 0.078128]),
         np.array([0.56544, 0.57024, 0.067786]),
         np.array([0.75009, 0.78412, 0]),
         np.array([0.74225, 0.56827, 0.076029]),
         np.array([0.74826, 0.14358, 0.030382]),
         np.array([0.56976, 0.13686, 0.049978]),
         np.array([0.74421, 0.35752, 0.04484]),
         np.array([0.56473, 0.35168, 0.095859])]

index = 'uav4'
for i in range(uav[index].shape[1]):
    for j in depot:
        if (uav[index][:, i] == j).all():
            print(j)

length = 0.
cap = 0.
for i in range(uav[index].shape[1]-1):
    if (uav[index][:, i+1] == np.array([0.57056, 0.78299, 0.078128])).all():
        length += np.sqrt(np.sum((uav[index][:, i+1] - uav[index][:, i])**2))
        cap += np.sqrt(np.sum((uav[index][:, i+1] - uav[index][:, i])**2))
        if cap >= 300/140:
            print('error')
        cap = 0
    else:
        length += np.sqrt(np.sum((uav[index][:, i+1] - uav[index][:, i])**2))
        cap += np.sqrt(np.sum((uav[index][:, i+1] - uav[index][:, i])**2))
        if cap >= 300/140:
            print('error')
print(length)
##
n_agent = 4
n_depot = 16
# run_20221110T195028
# 多机场景下训练
#os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\tests\\cvrp_30\\多机多起点多depot无balance')
os.chdir('E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\video\\run_20221215T130054')
data = pd.read_csv('routine.csv', header=None)
graph = pd.read_csv('graph_info.csv', header=None)
agent = pd.read_csv('agent_all.csv', header=None)
data = np.array(data)
graph = np.array(graph)
agent = np.array(agent)
colors = ['red', 'green', 'blue']
graph_size = 285
n_depot = 16
cal_num = 1

trajectory = [[i+n_depot] for i in range(n_agent)]
data_coord = []
for i in range(data.shape[1]):
    agent_idx = int(agent[0, i])
    trajectory[agent_idx].append(int(data[0, i]))
print(len(trajectory[0]),len(trajectory[1]),len(trajectory[2]),len(trajectory[3]))
for i in range(n_agent):
    for j in trajectory[i]:
        data_coord.append(graph[j])
##
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = torch.rand((1000, 3, 128,128)).to(device)
a_label = torch.randint(0, 10, (1000,)).to(device)
data = TensorDataset(a, a_label)
dataloader = DataLoader(data, 10, True)
if next(iter(dataloader))[0].is_cuda:
    print('gpu')
else:
    print(gpu)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # batch x 16 x 62
        self.conv1 = nn.Conv2d(3, 16, 6, 2)
        self.pool1 = nn.MaxPool2d(5, 2)
        #self.conv2 = nn.Conv2d(16, 6, 5, 2)
        self.fc1 = nn.Linear(16 * 29 * 29, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, 16 * 29 * 29)
        return self.fc1(x)


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    running_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        running_loss += loss
    print(epoch, running_loss)
##
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.rand((1000, 3, 128, 128))
y = torch.randint(0, 10, (1000,))
data = TensorDataset(x, y)
dataloader = DataLoader(data, 10, True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 6, 2)
        self.fc1 = nn.Linear()

##
import torch
import torch.nn as nn

# 定义三维数据（batch_size, sequence_length, hidden_size）
batch_size = 4
sequence_length = 5
hidden_size = 10
inputs = torch.randn(batch_size, sequence_length, hidden_size)

# Layer Normalization
layer_norm = nn.BatchNorm1d(sequence_length)
outputs = layer_norm(inputs)

# 打印结果
print("输入数据：")
print(inputs.shape)
print("Layer Normalization后的输出数据：")
print(outputs.shape)
##
from xgboost import XGBClassifier
a = XGBClassifier()