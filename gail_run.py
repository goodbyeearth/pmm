
from stable_baselines.gail import ExpertDataset
from pmm_gail_model import GAIL
from my_policies import CustomPolicy
from stable_baselines import PPO2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pommerman
from pommerman import agents

import sys

from my_cmd_utils import my_arg_parser
from my_policies import CustomPolicy
import numpy as np
from utils import featurize
import random
from tqdm import tqdm

def data_load():
    '''LOAD EXPERT DATASET'''
    print("LOAD %d DATASET data_path=%s" % (args.num_traj,args.data_path) )
    dataset = ExpertDataset(expert_path=args.data_path, traj_limitation=args.num_traj, verbose=1)
    print("=====> LOAD DATASET ok!")
    return dataset

def gail_train():
    '''LOAD DATASET'''
    dataset = data_load()
    print("INITIALIZING MODEL")
    print("MODEL USE POLICY policy_type=%s" % args.policy_type)
    policy_type = args.policy_type
    if args.policy_type == 'CustomPolicy':
        policy_type = CustomPolicy

    '''USING GAIL'''
    print("INITIALIZING GAIL PARAMETERS")
    print(" policy=%s \n tensorboard_log=%s " % (args.policy_type,args.log_path))
    model = GAIL(policy_type, 'PommeFFACompetition-v4', dataset, verbose=1,
                 full_tensorboard_log=True, tensorboard_log=args.log_path)
    print("=====> GAIL INIT OK")

    print("START LEARN num_timesteps=%f" % args.num_timesteps)
    model.learn(total_timesteps=args.num_timesteps)
    print("=====> GAIL LEARN OK")

    '''SAVE MODEL'''
    if args.save_path is not None:
        print("SAVE MODEL save_path=%s" % args.save_path)
        model.save(args.save_path)


def _test():
    '''LOAD MODEL'''
    print("LOAD MODEL load_path=%s" % args.load_path)
    model = GAIL.load(args.load_path)

    '''TEST MODEL'''
    from utils import featurize
    agent_list = [
                # agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
            ]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    print("=====> TEST ENV MAKE OK")
    for i_episode in range(1):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = env.act(obs)
            act, _states = model.predict(featurize(obs[3]))
            action[3] = int(act)
            obs, rewards, done, info = env.step(action)
            # print(action[3])
        print('Episode {} finished'.format(i_episode))
        print(rewards)
        print(info)
    env.close()

if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    if not args.play:
        print("IN TRAIN")
        gail_train()
    else:
        print("IN TEST")
        _test()

    # env_id = 'PommeFFACompetition-v4'



    '''单一环境'''
    # agent_list = [
    #             # agents.SimpleAgent(),
    #             agents.RandomAgent(),
    #             agents.BaseAgent(),
    #             agents.SimpleAgent(),
    #             agents.SimpleAgent()
    #         ]
    # env = pommerman.make(env_id, agent_list)
    # print("=====> env make ok!")

# def _pretrain(env,dataset):
#     """读取数据"""
#     dataset = data_load()
#     policy_type = args.policy_type
#     if args.policy_type == 'CustomPolicy':
#         policy_type = CustomPolicy
#
#     '''多线程设置'''
#     from my_subproc_vec_env import SubprocVecEnv
#     config = tf.ConfigProto()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     config.gpu_options.allow_growth = True
#
#     def make_envs(env_id):
#         def _thunk():
#             agent_list = [
#                 agents.SimpleAgent(),
#                 agents.SimpleAgent(),
#                 agents.SimpleAgent(),
#                 agents.BaseAgent()
#             ]
#             env = pommerman.make(env_id, agent_list)
#             return env
#         return _thunk
#
#     num_envs = multiprocessing.cpu_count()
#     envs = [make_envs(args.env_id) for _ in range(num_envs)]
#     env = SubprocVecEnv(envs)
#     print("=====> 多环境 make ok!")
#
#     '''使用PPO2进行 Bhevaral Cloning'''
#     model = PPO2(policy_type, env)
#     model.pretrain(dataset)
#
#     if args.save_path is not None:
#         model.save(args.save_path)

