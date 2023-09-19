#!/usr/bin/env python
import sys
sys.path.append('/home/work/风行电力交易/elec_trade/xd_test/other_file/MDAM-retry')
import os
import json
import copy
import time
import pprint as pp
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger


from options import get_options
from torch.utils.data import DataLoader
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel

from utils import torch_load_cpu, load_problem


def run(opts):
    """
    initialize training framework
    args:
        opts: parameter configuration
    returns:
    """
    pp.pprint(vars(opts))
    torch.manual_seed(opts.seed)

    tb_logger = None

    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    problem = load_problem(opts.problem)
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        # param dict
        load_data = torch_load_cpu(load_path)

    model_class = {
        'attention': AttentionModel
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        n_agent=opts.n_agent
    ).to(opts.device)

    # 是否有多个gpu供并行
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_ = get_inner_model(model)
    # **model_.state_dict()和**load_data.get('model', {})是key一样的dict，将前面的value换成后面对应key的value
    # 这里model和model_共享id（指针）
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # 只有第一个params有用
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
           [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # learning rate的decay，每个epoch的learning rate为learning_rate*(lr_decay**epoch)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # filename!=None: 使用外部数据（generate_data.py生成）
    # else: data = [{'loc': tensor.shape = (graph_size+n_agent, 3),
    #               'demand': tensor.shape = (graph_size+n_agent),
    #               'depot': tensor.shape = (n_depot)}] * val_size
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
        distribution=opts.data_distribution, opts=opts)
    
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            problem,
            tb_logger,
            opts
        )
            









if __name__ == "__main__":
    run(get_options())
