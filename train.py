import os
import time
import numpy as np
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    """
    if train in parallel, update model in a different way
    args:
        model: torch model
    returns:
    """
    return model.module if isinstance(model, DataParallel) else model

def predict_path(model, dataset, opts, save_result=False):
    set_decode_type(model, "beam")
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, pi, agent_all = model(bat, opts=opts, return_pi=True)
            return cost.data, pi.data, agent_all.data

    cost, pi, agent = None, None, None
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
        cost_, pi_, agent_all_ = eval_model_bat(bat)
        if cost == None:
            cost, pi, agent_all = cost_, pi_, agent_all_
        else:
            cost = torch.cat([cost, cost_], 0)
            pi = torch.cat([pi, pi_], 1)
            agent_all = torch.cat([agent_all, agent_all_], 1)
    pi, agent_all = pi.numpy(), agent_all.numpy()

    if save_result:
        if not os.path.exists(opts.test_absolute_dir):
            os.makedirs(opts.test_absolute_dir)

        np.savetxt('{}\\routine.csv'.format(opts.test_absolute_dir), pi, delimiter=',')
        np.savetxt('{}\\agent_all.csv'.format(opts.test_absolute_dir), agent_all, delimiter=',')
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost

def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost

def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    # 选点的时候选择softmax概率最大的那个
    set_decode_type(model, "greedy")
    # model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), opts=opts)
            return cost.data.cpu()



    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    """
    training framework
    args:
        model: torch model
        optimizer: torch optimizer
        baseline(class): baseline used to calculate the baseline value in reinforce loss
        lr_scheduler: learning rate decay
        epoch(int): current epoch
        val_dataset(class):
        problem(class): problem class
        tb_logger: tensorboard_logger
        opts: parameter configuration
    returns:

    """
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    # epoch_size每个epoch多少graph
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    
    # TODO: 什么时候rolloutbaseline什么时候warmupbaseline
    # 开始的warmupbaseline: training_dataset[i]是一个
    # dict={'loc': tensor.shape = (graph_size+n_agent, 3),
    #       'demand': tensor.shape = (graph_size+n_agent),
    #       'depot': tensor.shape = (n_depot)}
    # 共epoch_size个
    # 后面的rolloutbaseline: training_dataset[i]是一个
    # dict={'data':{'loc': tensor.shape = (graph_size+n_agent, 3),
    #               'demand': tensor.shape = (graph_size+n_agent),
    #               'depot': tensor.shape = (n_depot)}
    #       'baseline': }
    # 共epoch_size个
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution, opts=opts))

    # 一个迭代器，每个是一个
    # dict={'loc': tensor.shape = (batch_size, graph_size+n_agent, 3),
    #       'demand': tensor.shape = (batch_size, graph_size+n_agent),
    #       'depot': tensor.shape = (batch_size, n_depot)}
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    model.train()
    # sample action from distribution when training
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1
        lr_scheduler.step(epoch)



    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if epoch != 0 and ((opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
            (opts.n_epochs - opts.record_epoch >= 1 and epoch >= opts.record_epoch)):
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),  # model.state_dict(): 一个ordereddict（和dict类似，是key:value）key为网络的参数名称，key为对应的值（torch.tensor）
                'optimizer': optimizer.state_dict(),  # {'state': {}, 'param_groups': [{'lr': 0.0001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]}]}
                'rng_state': torch.get_rng_state(),  # 保存随机数生成器的状态
                'cuda_rng_state': torch.cuda.get_rng_state_all(),  # 保存gpu的随机数生成器的状态
                'baseline': baseline.state_dict()  # 保存baseline的参数
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):

    optimizer.zero_grad()
    # if warmup:
    # x: dict = {'loc': tensor.shape = (batch_size, graph_size+n_agent, 3),
    #            'demand': tensor.shape = (batch_size, graph_size+n_agent),
    #            'depot': tensor.shape = (batch_size, n_depot)}
    # bl_val = None
    # if rollout:
    # x: dict = {'loc': tensor.shape = (batch_size, graph_size+n_agent, 3),
    #            'demand': tensor.shape = (batch_size, graph_size+n_agent),
    #            'depot': tensor.shape = (batch_size, n_depot)}
    # TODO: bl_val in rollout baseline
    # bl_val =
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    cost, _log_p, pi, mask = model(x, opts, baseline, bl_val)

    ll = model._calc_log_likelihood(_log_p, pi, mask)
    if baseline != None:
        # ExponentialBaseline: 开始是cost的平均，后面是当前bl_val和当前cost平均的加权求和
        # rolloutbaseline: 模型输出的cost
        # bl_val: 1
        bl_val, _ = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        reinforce_loss = ((cost - bl_val) * ll).mean()
        loss = reinforce_loss
        loss.backward()
    # batch_size x n_paths
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    if grad_norms[0][0] != grad_norms[0][0]:
        optimizer.zero_grad()
        print("nan detected")
        return
    optimizer.step()
    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, 0, tb_logger, opts)

