import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='cvrp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training

    parser.add_argument('--bp_one_path', type=bool, default=False, help="bp for one path")
    parser.add_argument('--kl_loss', type=float, default=0.0, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_model', type=float, default=0.0001, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay per epoch')
    # , action = 'store_true'


    parser.add_argument('--n_epochs', type=int, default=3, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')


    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')
    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    # 'D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master-to-3d\\outputs\\cvrp_20\\run_20221027T191136\\epoch-99.pt',
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--n_paths', type=int, default=1, help='number of paths (number of decoders)')
    parser.add_argument('--n_EG', type=int, default=200, help='number of steps between EG')
    parser.add_argument('--test_size', type=int, default=1, help='number of graphs used to test the model')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Number of instances per batch during test')
    parser.add_argument('--test_dir', default='tests', help='Directory to record test results in')
    parser.add_argument('--n_agent', type=int, default=3, help='Number of agents')
    parser.add_argument('--dist_coef', type=int, default=1/3, help='coefficient of distance')
    parser.add_argument('--n_depot', type=int, default=3, help='Number of depots')
    parser.add_argument('--mean_distance', type=int, default=6, help='Number of depots')
    parser.add_argument('--safe_coef', type=float, default=10**(-6), help='coefficient used to Compensates for the lack of '
                                                                      'precision of single-precision floating-point '
                                                                      'numbers')
    '''
    # test_only
    parser.add_argument('--eval_only', type=bool, default=True, help='Set this value to only evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--load_path', default='E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance\\data\\epoch-199.pt',
                        help='Path to load model parameters and optimizer state from')
    parser.add_argument('--test_only', type=bool, default=True, help='whether to test the pretrained model')
    parser.add_argument('--val_dataset', type=str, default='data/test_data.pkl', help='Dataset file to use for validation')

    '''
    # train_only
    parser.add_argument('--eval_only', type=bool, default=False, help='Set this value to only evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--load_path', default=None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--test_only', type=bool, default=False, help='whether to test the pretrained model')
    #parser.add_argument('--load_path', default='D:\\xd\\project\\Combinatorial-Optimization-and-Artificial-Intelligence\\MDAM-master\\MDAM-master-to-3d-0\\outputs\\cvrp_30\\run_20221110T222210\\epoch-99.pt',help='Path to load model parameters and optimizer state from')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    #'''
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    if not opts.test_only:
        opts.save_dir = os.path.join(
            opts.output_dir,
            "{}_{}".format(opts.problem, opts.graph_size),
            opts.run_name
        )
    else:
        opts.save_dir = os.path.join(
            opts.test_dir,
            "{}_{}".format(opts.problem, opts.graph_size),
            opts.run_name
        )
    opts.test_absolute_dir = os.path.join(
        'E:\\xd\\project\\use_drl_to_solve_vrp\\MDAM-master\\MDAM-master-to-multi-depot0-unbalance',
        opts.test_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
