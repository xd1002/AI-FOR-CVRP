import sys

sys.path.append('C:\\Users\\xd1002\\Desktop\\AI-FOR-CVRP')
import os
import json
import logging
import torch

from options import get_options
from train import predict_path
from nets.attention_model import AttentionModel
from nets.attention_model_search import AttentionModelBeamSearch

from utils import torch_load_cpu, load_problem


def run(opts):
    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    opts.device = torch.device("cpu")

    problem = load_problem(opts.problem)
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        # param dict
        load_data = torch_load_cpu(load_path)

    model_class = {
        'attention': AttentionModel,
        "attention_beam": AttentionModelBeamSearch
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        problem,
        opts
    ).to(opts.device)

    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size,
        filename=opts.val_dataset, distribution=opts.data_distribution, opts=opts)
    predict_path(model, val_dataset, opts)


if __name__ == "__main__":
    run(get_options())