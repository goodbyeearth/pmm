
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

def _extra():
    '''Simple function to bootstrap a game.

           Use this as an example to secdt up your training env.
        '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12345),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v1', agent_list)
    print(env.observation_space)
    print(env.action_space)

    # Run the episodes just like OpenAI Gym
    actions = []
    observations = []
    rewards = []
    episode_returns = []
    episode_starts = []
    print("提取专家数据数量: num_traj_e=%d" % args.num_traj_e)
    print("随机提取每回合 %d% 的数据" % args.rand)
    for i_episode in tqdm(range(args.num_traj_e)):
        state = env.reset()
        flag = True
        done = False
        while not done:
            if flag:
                episode_starts.append(True)
            else:
                episode_starts.append(False)
            # env.render()
            acts = env.act(state)
            state, reward, done, info = env.step(acts)
            if not env._agents[3].is_alive:
                done = True
                reward = [0, 0, 0, -1]
            # 随机存取
            rand = random.randint(0,100)
            if rand < args.rand or flag:
                obs = featurize(state[3])
                observations.append(obs)
                rewards.append(reward[3])
                actions.append(acts[3])
            flag = False
        print('Episode {} finished'.format(i_episode))
        episode_returns.append(reward[3])
    env.close()

    numpy_dict = {
        'actions': np.array(actions).reshape(-1, 1),
        'obs': np.array(observations),
        'rewards': np.array(rewards),
        'episode_returns': np.array(episode_returns),
        'episode_starts': np.array(episode_starts)
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)
    print("保存专家数据到: data_path=%s" % args.data_path)
    np.savez(args.data_path, **numpy_dict)

def data_load():
    '''加载 expert dataset'''
    print("正在读取%d条专家数据, 数据路径为: data_path=%s" % (args.num_traj,args.data_path) )
    dataset = ExpertDataset(expert_path=args.data_path, traj_limitation=args.num_traj, verbose=1)
    print("=====> 加载专家数据 ok!")
    return dataset

def gail_train():
    '''读取数据'''
    dataset = data_load()
    print("正在初始化model")
    print("初始化model的policy为: policy_type=%s" % args.policy_type)
    policy_type = args.policy_type
    if args.policy_type == 'CustomPolicy':
        policy_type = CustomPolicy

    '''使用GAIL'''
    print("初始化GAIL参数为:")
    print(" policy=%s \n tensorboard_log=%s " % (args.policy_type,args.log_path))
    model = GAIL(policy_type, 'PommeFFACompetition-v4', dataset, verbose=0,
                 full_tensorboard_log=True, tensorboard_log=args.log_path)
    print("=====> gail init ok!")
    print("开始训练model, num_timesteps=%f" % args.num_timesteps)
    model.learn(total_timesteps=args.num_timesteps)
    print("=====> gail learn ok!")
    '''保存模块'''
    print("保存模型到: save_path=%s" % args.save_path)
    if args.save_path is not None:
        model.save(args.save_path)


def _test():
    '''加载模块'''
    print("加载模型: load_path=%s" % args.load_path)
    model = GAIL.load(args.load_path)

    '''测试模块 未完成'''
    from utils import featurize
    agent_list = [
                # agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
                agents.SimpleAgent(),
            ]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    print("=====> test env make ok!")
    for i_episode in range(1):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = env.act(obs)
            act, _states = model.predict(featurize(obs[3]))
            action[3] = int(act)
            obs, rewards, done, info = env.step(action)
        print('Episode {} finished'.format(i_episode))
        print(rewards)
        print(info)
    env.close()

if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    if args.extra:
        print("正在进行专家数据提取, 保存路径为: %s" % args.data_path)
        _extra()
    if not args.play:
        print("进入训练模块")
        gail_train()
    else:
        print("进入测试模块")
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

