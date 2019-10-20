from stable_baselines.common.cmd_util import arg_parser


def my_arg_parser():
    parser = arg_parser()

    parser.add_argument('--env', help='environment ID', type=str, default='PommeRadioCompetition-v2')
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel.',
                        default=None, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e3),
    parser.add_argument('--nsteps', type=int, default=5)
    parser.add_argument('--policy_type',
                        help='policy type (MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, '
                             'CnnLstmPolicy, CnnLnLstmPolicy)',
                        default='MlpPolicy')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--data_path', help='Path to dataset', default=None, type=str)
    parser.add_argument('--log_path', help='Path to log', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model from', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--num_traj', type=int, default=-1)
    return parser
