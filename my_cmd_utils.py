from stable_baselines.common.cmd_util import arg_parser


def my_arg_parser():
    parser = arg_parser()

    parser.add_argument('--env', help='environment ID', type=str, default='PommeRadioCompetition-v2')
    # parser.add_argument('--env_type',
    #                     help='type of environment, used when the environment type cannot be automatically determined',
    #                     type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--nsteps', type=int, default=5),
    parser.add_argument('--policy_type',
                        help='policy type (MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, '
                             'CnnLstmPolicy, CnnLnLstmPolicy)',
                        default='MlpPolicy')
    # parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    # parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env',
                        help='Number of environment copies being run in parallel.',
                        default=None, type=int)
    # parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Path to log', default=None, type=str)
    # TODO:以下俩参数要不要保留呢
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--play', default=False, action='store_true')
    # parser.add_argument('--extra_import', help='Extra module to import to access external environments', type=str,
    #                    default=None)
    return parser
