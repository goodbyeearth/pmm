from my_record_expert import generate_expert_traj_v2
import pommerman
from pommerman import agents
from stable_baselines.common.vec_env import CloudpickleWrapper

import multiprocessing
import os

import numpy as np


def merge_data(dir_path, final_data_path):
    files_list = os.listdir(dir_path)
    flag = True
    for f_name in files_list:
        f_path = dir_path + f_name
        sub_data = np.load(f_path)
        if flag:   # 第一次迭代执行
            actions = sub_data['actions']
            obs = sub_data['obs']
            rewards = sub_data['rewards']
            episode_returns = sub_data['episode_returns']
            flag = False
        else:
            actions = np.concatenate((actions, sub_data['actions']))
            obs = np.concatenate((obs, sub_data['obs']))
            rewards = np.concatenate((rewards, sub_data['rewards']))
            episode_returns = np.concatenate((episode_returns, sub_data['episode_returns']))
    numpy_dict = {
        'actions': actions,
        'obs': obs,
        'rewards': rewards,
        'episode_returns': episode_returns,
    }
    print("以下数据将写入 {}".format(final_data_path))
    for key, val in numpy_dict.items():
        print(key, val.shape)
    np.savez(final_data_path, **numpy_dict)

def generate_expert_data(env_id, n_process=None, n_episodes=250):
    n_process = n_process or multiprocessing.cpu_count()
    envs = [make_envs(env_id) for _ in range(n_process)]
    ps = []
    for i, env in zip(range(n_process), envs):
        prefix_path = './dataset_test/e' + str(n_episodes) + '_p' + str(i) + '_a'
        ps.append(multiprocessing.Process(target=worker, args=(CloudpickleWrapper(env), n_episodes, prefix_path)))
    print('开始取数据，进程数量：', n_process)
    for p in ps:
        p.start()


def worker(env_fn_wrapper, n_episodes=100, prefix_path=None):
    assert prefix_path is not None
    env = env_fn_wrapper.var()
    record_idx_list = [0, 1, 2, 3]

    save_path_list = [prefix_path + str(idx) for idx in record_idx_list]

    print('总回合数：{}, 目标智能体编号：{}'.format(n_episodes, record_idx_list))
    # 开始爬取并存储数据
    generate_expert_traj_v2(env, record_idx_list, save_path_list, n_episodes=n_episodes)


def make_envs(env_id):
    def _thunk():
        agent_list = [
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]
        env = pommerman.make(env_id, agent_list)
        return env
    return _thunk


if __name__ == '__main__':
    merge_data('./dataset_test/', './final_data_test/final_data')