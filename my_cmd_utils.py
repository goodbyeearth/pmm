from stable_baselines.common.cmd_util import arg_parser


def my_arg_parser():
    parser = arg_parser()

    # 模式
    parser.add_argument('--train', help='训练强化学习算法', default=False, action='store_true')
    parser.add_argument('--generate_data', help='获取并存储专家数据', default=False, action='store_true')   # 生成专家数据相关
    parser.add_argument('--merge_data', help='合并专家数据', default=False, action='store_true')  # 生成专家数据相关
    parser.add_argument('--pre_train', help='监督学习训练', default=False, action='store_true')
    parser.add_argument('--play', help='跑训练好的算法', default=False, action='store_true')
    parser.add_argument('--using_PGN', help='新建网络', default=False, action='store_true')
    parser.add_argument('--evaluate', help='保存这次的训练参数', default=False, action='store_true')

    parser.add_argument('--env', help='环境名称', type=str, default='PommeRadioCompetition-v2')
    parser.add_argument('--num_env',
                        help='（在强化学习算法训练的时候）并行环境数量，默认根据 CPU 来选取数量',
                        default=None, type=int)

    parser.add_argument('--alg', help='强化学习算法名称', type=str, default='ppo2')
    parser.add_argument('--policy_type',
                        help="策略类型，包括自定义的，以及 stable_baseline提供的 (MlpPolicy, MlpLstmPolicy, "
                             "MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy)",
                        default='MlpPolicy')

    parser.add_argument('--num_timesteps', help='强化学习算法训练所用步数', type=float, default=1e6),
    parser.add_argument('--nsteps', type=int, default=5),
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)

    parser.add_argument('--data_episode', type=int, default=None)     # 生成专家数据相关

    # 路径
    parser.add_argument('--save_path', help='保存强化学习训练好的模型的路径', default=None, type=str)
    parser.add_argument('--load_path', help='加载强化学习训练好的模型的路径', default=None, type=str)
    parser.add_argument('--log_path', help='保存强化学习训练日志的路径', default=None, type=str)
    parser.add_argument('--expert_path', help='保存专家数据的路径', default=None, type=str)   # 生成专家数据相关


    return parser
