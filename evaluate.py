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
# from tensorboard_logger import Logger as TbLogger


from options import get_options
from torch.utils.data import DataLoader
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel

from utils import torch_load_cpu, load_problem


def run(opts):

    torch.manual_seed(opts.seed)

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
        n_EG=opts.n_EG,
        n_agent=opts.n_agent
    ).to(opts.device)
    
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
     
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution, opts=opts)
    
    validate(model, val_dataset, opts)            


if __name__ == "__main__":
    run(get_options())
