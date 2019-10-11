import pommerman
from pommerman import agents

import sys
import os

from importlib import import_module
import multiprocessing

import tensorflow as tf

from my_cmd_utils import my_arg_parser
from my_subproc_vec_env import SubprocVecEnv
from my_policies import CustomPolicy


# TODO：加seed
def make_envs(env_id):
    def _thunk():
        agent_list = [
            # agents.SimpleAgent(),
            agents.RandomAgent(),
            agents.BaseAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]
        env = pommerman.make(env_id, agent_list)
        return env
    return _thunk


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = import_module('.'.join(['stable_baselines', alg, submodule]))
    return alg_module


def get_model_fn(alg):
    if alg == 'a2c':
        return get_alg_module(alg).A2C
    if alg == 'ppo2':
        return get_alg_module(alg).PPO2
    print("算法", alg, ' 未注册到run.get_model里')
    raise ImportError


def train():
    total_timesteps = int(args.num_timesteps)

    # 多线程设置
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs(args.env) for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    # 设置 policy_type
    policy_type = args.policy_type
    if policy_type == 'CustomPolicy':
        policy_type = CustomPolicy

    # 初始化 model，如 PPO2
    model_fn = get_model_fn(args.alg)
    model = model_fn(policy_type, env, verbose=1, tensorboard_log=args.log_path)

    # 预训练
    if args.pre_train:
        from my_dataset import ExpertDataset
        # assert args.expert_path is not None
        # 加载数据 TODO: 路径使用参数, 设置 traj_limitation
        print("开始加载专家数据...")
        dataset = ExpertDataset(expert_path='./dataset_test/1_expert_agent_1_100.npz')  #  traj_limitation=100
        # 开始与训练 TODO: 设置 epoch 数量
        print("开始在{}模型上进行预训练...\nPolicy type:{}".format(args.alg, args.policy_type))
        model.pretrain(dataset=dataset, n_epochs=100)
        # TODO: 要不要保存
        model.save('./pretrain_model_test/pretrain_model.zip')

    print('开始利用强化学习训练{}模型，进程数：{}, policy type:{}'
          .format(args.alg, num_envs, args.policy_type))
    # TODO: 可以加个 call back
    model.learn(total_timesteps=total_timesteps,
                seed=args.seed)

    if args.save_path:
        model.save(save_path=args.save_path)

    env.close()


def play():
    if not args.load_path:
        print('在 play 模式下，务必添加参数 --load_path')
        raise ValueError

    from utils import featurize

    model_fn = get_model_fn(args.alg)
    model = model_fn.load(args.load_path)
    env_fn = make_envs(args.env)
    env = env_fn()

    # 设置训练的 agent 的 index
    train_idx = 1
    env.set_training_agent(train_idx)

    def get_all_actions():
        feature = featurize(obs[train_idx])
        action, _states = model.predict(feature)
        action = (action, 0, 0)
        some_actions = env.act(obs)  # 不包含我的 agent
        # 如果其他智能体动作不是元组（只有单一动作），改成元组
        for agent_idx in range(3):
            if not isinstance(some_actions[agent_idx], tuple):
                some_actions[agent_idx] = (some_actions[agent_idx], 0, 0)
        some_actions.insert(train_idx, action)  # 把我的 agent 的动作也加进来

        return some_actions

    for episode in range(5):
        obs = env.reset()
        is_dead = False  # 标志我的智能体死没死
        for i in range(1000):
            all_actions = get_all_actions()
            obs, rewards, done, info = env.step(all_actions)
            if not is_dead and not env._agents[env.training_agent].is_alive:
                print("My agent is dead. ~.~")
                is_dead = True
                break  # 死了重来~~

            if done:
                break
            env.render()
    env.close()


def generate_expert_data():
    from my_record_expert import generate_expert_traj

    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
    env = pommerman.make(args.env, agent_list)

    # 记得设置相关参数
    n_episodes = 10
    record_idx_list = [2, 3]

    # TODO: 路径使用参数
    # 设置 dataset 存储路径，每个 agent 的数据存放在不同的文件里
    save_path_list = ['./dataset_test/expert_agent_' + str(idx) for idx in record_idx_list]

    print('总回合数：{}, 目标智能体编号：{}'.format(n_episodes, record_idx_list))
    # 开始爬取并存储数据
    generate_expert_traj(env, record_idx_list, save_path_list, n_episodes=n_episodes)


if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    print("环境：", args.env)

    # 爬取并存储专家数据，记得在函数内部设置回合数等等
    if args.generate_data:
        generate_expert_data()

    # 训练一波
    if args.train:
        train()

    # 跑一波看看效果
    if args.play:
        play()
