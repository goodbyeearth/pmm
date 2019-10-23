import pommerman
from pommerman import agents

import sys
import os

from importlib import import_module
import multiprocessing

import tensorflow as tf

from my_cmd_utils import my_arg_parser
from my_subproc_vec_env import SubprocVecEnv
from my_policies import CustomPolicy, CustomLSTM

from generate_data import generate_expert_data, merge_data

# TODO：加seed
def make_envs_0(env_id):
    def _thunk():
        agent_list = [
            agents.BaseAgent(),
            agents.SimpleAgent(),
            agents.RandomAgent(),
            agents.SimpleAgent(),
            # agents.SimpleAgent()
            # agents.RandomAgent(),
        ]
        env = pommerman.make(env_id, agent_list)
        env.set_training_agent(0)
        return env
    return _thunk

def make_envs_1(env_id):
    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            agents.BaseAgent(),
            agents.SimpleAgent(),
            agents.RandomAgent(),
            # agents.SimpleAgent()
            # agents.RandomAgent(),
        ]
        env = pommerman.make(env_id, agent_list)
        env.set_training_agent(1)
        return env
    return _thunk


def make_envs_2(env_id):
    def _thunk():
        agent_list = [
            agents.RandomAgent(),
            agents.SimpleAgent(),
            agents.BaseAgent(),
            agents.SimpleAgent(),
            # agents.SimpleAgent()
            # agents.RandomAgent(),
        ]
        env = pommerman.make(env_id, agent_list)
        env.set_training_agent(2)
        return env
    return _thunk


def make_envs_3(env_id):
    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            agents.RandomAgent(),
            agents.SimpleAgent(),
            agents.BaseAgent(),
            # agents.SimpleAgent()
            # agents.RandomAgent(),
        ]
        env = pommerman.make(env_id, agent_list)
        env.set_training_agent(3)
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
    # print("算法", alg, ' 未注册到run.get_model里')
    raise ImportError


def train():
    total_timesteps = int(args.num_timesteps)

    # 多线程设置
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs_0(args.env) for _ in range(num_envs//4)] + \
           [make_envs_1(args.env) for _ in range(num_envs//4)] + \
           [make_envs_2(args.env) for _ in range(num_envs//4)] + \
           [make_envs_3(args.env) for _ in range(num_envs//4)]

    env = SubprocVecEnv(envs)

    # 设置 policy_type
    policy_type = args.policy_type
    if policy_type == 'CustomPolicy':
        policy_type = CustomPolicy

    # policy_type = CustomLSTM

    # 初始化 model，如 PPO2
    model_fn = get_model_fn(args.alg)
    model = model_fn(policy_type, env, verbose=1, tensorboard_log=args.log_path)

    # 预训练
    if args.pre_train:
        from my_dataset import ExpertDataset
        # assert args.expert_path is not None
        # 加载数据
        print("loading data...")
        # dataset = ExpertDataset(expert_path='./dataset_test/e200_p1_a0.npz', batch_size=num_envs)
        dataset = ExpertDataset(expert_path='./final_data_test/7w.npz')  #  traj_limitation 只能取默认-1
        # 开始与训练
        print("pretrain in {}...\nPolicy type:{}".format(args.alg, args.policy_type))
        model.pretrain(dataset=dataset, n_epochs=30)
        model.save('./pretrain_model_test/pretrain_model.zip')

    print('training in RL, alg:{}, env:{}, policy type:{}'
          .format(args.alg, num_envs, args.policy_type))
    # TODO: 可以加个 call back
    model.learn(total_timesteps=total_timesteps,
                seed=args.seed)

    if args.save_path:
        model.save(save_path=args.save_path)

    env.close()


def play():
    if not args.load_path:
        print('play mode,must add --load_path')
        raise ValueError

    from utils import featurize

    model_fn = get_model_fn(args.alg)
    model = model_fn.load(args.load_path)
    # env_fn = make_envs(args.env)
    # env = env_fn()
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.BaseAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.RandomAgent(),
    ]
    env = pommerman.make(args.env, agent_list)

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

    total_eps = 15
    lost_eps = 0
    for episode in range(total_eps):
        obs = env.reset()
        is_dead = False  # 标志我的智能体死没死
        for i in range(1000):
            all_actions = get_all_actions()
            obs, rewards, done, info = env.step(all_actions)
            # if not is_dead and not env._agents[train_idx].is_alive:
            # input(obs[0]['alive'])
            if not is_dead and ((train_idx + 10) not in obs[0]['alive']):
                print("My agent is dead. ~.~")
                lost_eps += 1
                break  # 死了重来~~

            if done:
                break
            env.render()
    win_rate = (total_eps - lost_eps) / total_eps
    print("win rate: ", win_rate)
    env.close()


if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)
    print("environment: ", args.env)

    # 爬取并存储专家数据
    if args.generate_data:
        generate_expert_data(args.env, n_episodes=5000)

    if args.merge_data:
        merge_data('./dataset_test/', './final_data_test/final_data')

    # 训练一波
    if args.train:
        train()

    # 跑一波看看效果
    if args.play:
        play()
