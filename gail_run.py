import gym

from stable_baselines.gail import ExpertDataset, generate_expert_traj
from pmm_gail_model import GAIL
# from stable_baselines import GAIL
from my_policies import CustomPolicy
from stable_baselines import PPO2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import multiprocessing
import pommerman
from pommerman import agents
import tensorflow as tf



if __name__ == '__main__':
    #
    # env_id = 'PommeFFACompetition-v4'
    #
    # '''多线程设置'''
    # # from my_subproc_vec_env import SubprocVecEnv
    # # config = tf.ConfigProto()
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # # config.gpu_options.allow_growth = True
    # #
    # # def make_envs(env_id):
    # #     def _thunk():
    # #         agent_list = [
    # #             # agents.SimpleAgent(),
    # #             agents.RandomAgent(),
    # #             agents.BaseAgent(),
    # #             agents.SimpleAgent(),
    # #             agents.SimpleAgent()
    # #         ]
    # #         env = pommerman.make(env_id, agent_list)
    # #         return env
    # #     return _thunk
    # #
    # # num_envs = multiprocessing.cpu_count()
    # # envs = [make_envs(env_id) for _ in range(num_envs)]
    # # env = SubprocVecEnv(envs)
    # # print("=====> 多环境 make ok!")
    #
    # '''单一环境'''
    # # agent_list = [
    # #             # agents.SimpleAgent(),
    # #             agents.RandomAgent(),
    # #             agents.BaseAgent(),
    # #             agents.SimpleAgent(),
    # #             agents.SimpleAgent()
    # #         ]
    # # env = pommerman.make(env_id, agent_list)
    # # print("=====> env make ok!")
    #
    # '''加载 expert dataset'''
    # dataset = ExpertDataset(expert_path='dataset/1_expert_agent_1_100.npz', traj_limitation=-1, verbose=1)
    # print("=====> 加载专家数据 ok!")
    #
    # '''使用GAIL'''
    # model = GAIL(CustomPolicy, env_id, dataset, verbose=1)
    # print("=====> gail init ok!")
    # model.learn(total_timesteps=10000)
    # print("=====> gail learn ok!")
    #
    # '''使用PPO2进行 Bhevaral Cloning'''
    # # model = PPO2(CustomPolicy,env)
    # # model.pretrain(dataset)
    #
    # '''保存模块'''
    # model.save("model/gail_pmm")
    #
    # # env.close()
    # del model # remove to demonstrate saving and loading

    '''加载模块'''
    model = GAIL.load("model/gail_pmm")

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
    obs = env.reset()
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

