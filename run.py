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
    env_id = args.env

    # 多线程设置
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs(env_id) for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    model_fn = get_model_fn(args.alg)

    policy_type = args.policy_type
    if policy_type == 'CustomPolicy':
        policy_type = CustomPolicy

    model = model_fn(policy_type, env, verbose=1, tensorboard_log=args.log_path)
    print('在环境 {} 上训练 {} 模型，进程数：{}, policy type:{}'
          .format(env_id, args.alg, num_envs, args.policy_type))
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

    def get_all_actions(obs, train_agent_idx):
        feature = featurize(obs[train_idx])
        action, _states = model.predict(feature)
        action = tuple(action)
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
            all_actions = get_all_actions(obs, train_idx)
            obs, rewards, dones, info = env.step(all_actions)
            if not is_dead and not env._agents[env.training_agent].is_alive:
                print("My agent is dead. ~.~")
                is_dead = True
                break  # 死了重来~~
            env.render()
    env.close()


if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    if not args.play:
        train()
    else:
        play()
