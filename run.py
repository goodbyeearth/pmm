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


def train(args):
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

    return model, env


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


def main(args):
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    # TODO:logger配置

    model, env = train(args)
    env.close()

    if args.save_path:
        model.save(args.save_path)

    return model


if __name__ == '__main__':
    main(sys.argv)
