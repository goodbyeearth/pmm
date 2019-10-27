import numpy as np
from gym import spaces
from tqdm import tqdm

import random


def generate_expert_traj_v1(env=None, agent_idx_list=None, save_path_list=None, n_episodes=100):
    from utils import featurize, get_feature_space, get_action_space
    """获得所有回合的所有数据"""
    # Check
    assert isinstance(get_feature_space(), spaces.Box)
    assert isinstance(get_action_space(), spaces.Discrete)
    assert env is not None
    assert agent_idx_list is not None
    assert save_path_list is not None

    n_record = len(agent_idx_list)         # 同时收集数据的智能体数目

    # 以下的所有列表中的每一个元素为一个子列表，每个自列表对应每个智能体收集的数据
    actions_list = [[] for _ in range(n_record)]
    observations_list = [[] for _ in range(n_record)]    # 保存处理后的 feature, 而不是 dict
    rewards_list = [[] for _ in range(n_record)]
    episode_returns_list = [np.zeros((n_episodes,)) for _ in range(n_record)]  # 该列表中的每个元素，长度为 n_episodes
    # episode_starts_list = [[] for _ in range(n_record)]

    ep_idx = 0    # 记录对战回合数
    obs = env.reset()
    # for i in range(n_record):
    #     episode_starts_list[i].append(True)
    # reward_sum_list = [0.0 for _ in range(n_record)]

    while ep_idx < n_episodes:
        # 每个智能体观察到的 obs[agent_idx] 经过特征化后，append 到各自的 observations（本身就是列表）里
        feature_list = []
        for agent_idx in agent_idx_list:
            # 死了就 append None
            if env._agents[agent_idx].is_alive:
                feature_list.append(featurize(obs[agent_idx]))   # 先保存 feature
            else:
                feature_list.append(None)

        all_action = env.act(obs)

        obs, reward, done, _ = env.step(all_action)    # obs 和 reward 均为长度为4的列表

        for i, agent_idx in zip(range(n_record), agent_idx_list):
            if env._agents[agent_idx].is_alive:  # 活着的才记录
                assert feature_list[agent_idx] is not None
                observations_list[i].append(feature_list[agent_idx])
                actions_list[i].append(all_action[agent_idx])
                rewards_list[i].append(reward[agent_idx])
                # episode_starts_list[i].append(done)
                # reward_sum_list[i] += reward[agent_idx]

        if done:
            obs = env.reset()
            for i, agent_idx in zip(range(n_record), agent_idx_list):
                # if env._agents[agent_idx].is_alive:  # 活着的才记录
                #     episode_returns_list[i][ep_idx] = reward_sum_list[i]
                episode_returns_list[i][ep_idx] = reward[agent_idx]

            # reward_sum_list = [0.0 for _ in range(n_record)]
            ep_idx += 1
            if ep_idx % 10 == 0:
                print('爬取数据中，当前回合：{}'.format(ep_idx))

    """
    在写入文件前，将单个智能体在所有回合里的数据保存为一个很大的 np.array
    a : [array([0, 0, 0]), array([1, 1, 1]), array([2, 2, 2]), array([3, 3, 3])]
    b = np.concatenate(a)
    --->  b : array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    b.reshape((-1, 3, ))
    --->  b: array([[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
    """
    for i in range(n_record):
        observations_list[i] = np.concatenate(observations_list[i]).reshape((-1,) + get_feature_space().shape)
        actions_list[i] = np.array(actions_list[i]).reshape((-1, 1))
        assert len(observations_list[i]) == len(actions_list[i])    # 确认一下
        rewards_list[i] = np.array(rewards_list[i])       # 列表变 array
        # episode_starts_list[i] = np.array(episode_starts_list[i][:-1])  # 最后一位不要，在最开始的时候已多补了第一位

    # 每个智能体的数据都包装成 numpy dict
    print("将产生 {} 个文件，每个文件对应一个智能体。".format(n_record))
    for i in range(n_record):
        numpy_dict = {
            'actions': actions_list[i],
            'obs': observations_list[i],
            'rewards': rewards_list[i],
            'episode_returns': episode_returns_list[i],
            # 'episode_starts': episode_starts_list[i]
        }
        print("以下数据将写入 {}".format(save_path_list[i]))
        for key, val in numpy_dict.items():
            print(key, val.shape)
        np.savez(save_path_list[i], **numpy_dict)
        print("以上数据写入完成。")
        print("====================================")

    env.close()


def generate_expert_traj_v2(env=None, agent_idx_list=None, save_path_list=None, n_episodes=50000):
    from utils import featurize, get_feature_space, get_action_space
    """每回合抽取一条的数据"""
    # todo: episode_start 这条数据有问题，暂时别用
    # Check
    assert isinstance(get_feature_space(), spaces.Box)
    assert isinstance(get_action_space(), spaces.Discrete)
    assert env is not None
    assert agent_idx_list is not None
    assert save_path_list is not None

    n_record = len(agent_idx_list)         # 同时收集数据的智能体数目

    # 以下为最终保存的数据，每回合只取1条数据
    actions_list = [[] for _ in range(n_record)]
    observations_list = [[] for _ in range(n_record)]    # 保存处理后的 feature, 而不是 dict
    rewards_list = [[] for _ in range(n_record)]
    episode_returns_list = [np.zeros((n_episodes,)) for _ in range(n_record)]  # 该列表中的每个元素，长度为 n_episodes
    # episode_starts_list = [[] for _ in range(n_record)]

    # 以下暂时保存一个回合内的所有数据，回合结束将被清空
    temp_actions_list = [[] for _ in range(n_record)]
    temp_observations_list = [[] for _ in range(n_record)]  # 保存处理后的 feature, 而不是 dict
    temp_rewards_list = [[] for _ in range(n_record)]
    # temp_episode_returns_list = [np.zeros((n_episodes,)) for _ in range(n_record)]  # 该列表中的每个元素，长度为 n_episodes
    # temp_episode_starts_list = [[] for _ in range(n_record)]

    ep_idx = 0    # 记录对战回合数
    obs = env.reset()
    # for i in range(n_record):
    #     temp_episode_starts_list[i].append(True)

    while ep_idx < n_episodes:
        # 每个智能体观察到的 obs[agent_idx] 经过特征化后，append 到各自的 observations（本身就是列表）里
        feature_list = []
        for agent_idx in agent_idx_list:
            # 死了就 append None
            if env._agents[agent_idx].is_alive:
                feature_list.append(featurize(obs[agent_idx]))   # 先保存 feature
            else:
                feature_list.append(None)

        all_action = env.act(obs)

        obs, reward, done, _ = env.step(all_action)    # obs 和 reward 均为长度为4的列表

        for i, agent_idx in zip(range(n_record), agent_idx_list):
            if env._agents[agent_idx].is_alive:  # 活着的才记录
                assert feature_list[agent_idx] is not None

                temp_observations_list[i].append(feature_list[agent_idx])
                temp_actions_list[i].append(all_action[agent_idx])
                temp_rewards_list[i].append(reward[agent_idx])
                # temp_episode_starts_list[i].append(done)

        if done:
            obs = env.reset()

            for i, agent_idx in zip(range(n_record), agent_idx_list):
                # 最后才给出 episode returns
                episode_returns_list[i][ep_idx] = reward[agent_idx]
                # 一回合里提一条数据
                sample_idx = random.randint(0, len(temp_rewards_list[i])-1)
                observations_list[i].append(temp_observations_list[i][sample_idx])
                actions_list[i].append(temp_actions_list[i][sample_idx])
                rewards_list[i].append(temp_rewards_list[i][sample_idx])
                # episode_starts_list[i].append(temp_episode_starts_list[i][sample_idx])
                # 清空当前智能体在当前回合的所有数据
                temp_observations_list[i] = []
                temp_actions_list[i] = []
                temp_rewards_list[i] = []
                # temp_episode_starts_list[i] = [True]          # todo:有问题

            ep_idx += 1
            if ep_idx % 10 == 0:
                print('爬取数据中，当前回合：{}'.format(ep_idx))

    """
    在写入文件前，将单个智能体在所有回合里的数据保存为一个很大的 np.array
    a : [array([0, 0, 0]), array([1, 1, 1]), array([2, 2, 2]), array([3, 3, 3])]
    b = np.concatenate(a)
    --->  b : array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    b.reshape((-1, 3, ))
    --->  b: array([[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
    """
    for i in range(n_record):
        observations_list[i] = np.concatenate(observations_list[i]).reshape((-1,) + get_feature_space().shape)
        actions_list[i] = np.array(actions_list[i]).reshape((-1, 1))
        assert len(observations_list[i]) == len(actions_list[i])    # 确认一下
        rewards_list[i] = np.array(rewards_list[i])       # 列表变 array
        # episode_starts_list[i] = np.array(episode_starts_list[i][:-1])  # 最后一位不要，在最开始的时候已多补了第一位

    # 每个智能体的数据都包装成 numpy dict
    print("将产生 {} 个文件，每个文件对应一个智能体。".format(n_record))
    for i in range(n_record):
        numpy_dict = {
            'actions': actions_list[i],
            'obs': observations_list[i],
            'rewards': rewards_list[i],
            'episode_returns': episode_returns_list[i],
            # 'episode_starts': episode_starts_list[i]
        }
        print("以下数据将写入 {}".format(save_path_list[i]))
        for key, val in numpy_dict.items():
            print(key, val.shape)
        np.savez(save_path_list[i], **numpy_dict)
        print("以上数据写入完成。")
        print("====================================")

    env.close()


def generate_expert_traj_v3(env=None, agent_idx_list=None, path_prefix=None, n_episodes=50, p_idx=None):
    """每回合抽取一条的数据，不 featurize"""
    # Check
    assert env is not None
    assert agent_idx_list is not None
    assert path_prefix is not None
    assert p_idx is not None

    n_record = len(agent_idx_list)         # 同时收集数据的智能体数目

    # 以下为最终保存的数据，每回合只取1条数据
    actions_list = [[] for _ in range(n_record)]
    observations_list = [[] for _ in range(n_record)]    # 保存处理后的 feature, 而不是 dict

    # 以下暂时保存一个回合内的所有数据，回合结束将被清空
    temp_actions_list = [[] for _ in range(n_record)]
    temp_observations_list = [[] for _ in range(n_record)]  # 保存处理后的 feature, 而不是 dict

    obs = env.reset()
    eps_idx = 0
    while eps_idx < n_episodes:
        all_action = env.act(obs)
        for i, agent_idx in zip(range(n_record), agent_idx_list):
            if env._agents[agent_idx].is_alive:
                temp_observations_list[i].append(obs[agent_idx])
                temp_actions_list[i].append(all_action[agent_idx])

        obs, reward, done, _ = env.step(all_action)    # obs 和 reward 均为长度为4的列表

        if done:
            obs = env.reset()
            for i, agent_idx in zip(range(n_record), agent_idx_list):
                # 一回合里提一条数据
                sample_idx = random.randint(0, len(temp_actions_list[i])-1)
                print(sample_idx)
                observations_list[i].append(temp_observations_list[i][sample_idx])
                actions_list[i].append(temp_actions_list[i][sample_idx])

                # 清空当前智能体在当前回合的所有数据
                temp_observations_list[i] = []
                temp_actions_list[i] = []
            eps_idx += 1
            if eps_idx%10 == 0:
                print('generating data, episode:', eps_idx)

    for i in range(n_record):
        assert len(observations_list[i]) == len(actions_list[i])    # 确认一下

    # 每个智能体的数据都包装成 numpy dict
    print("all data will write into {} files".format(n_record))
    for i in range(n_record):
        numpy_dict = {
            'actions': actions_list[i],
            'obs': observations_list[i],
        }
        save_path = path_prefix + 'agent_' + str(i) + '/proc_' + str(p_idx) + '_epis_' + str(n_episodes)
        np.savez(save_path, **numpy_dict)
        print("data_{} saved into {}".format(i, save_path))
        print("====================================")

    env.close()