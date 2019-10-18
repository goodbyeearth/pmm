import numpy as np
from utils import featurize
import random
from tqdm import tqdm
import pommerman
from pommerman import agents
import time

def _extra(rand,data_path,num_traj):
    '''Simple function to bootstrap a game.
           Use this as an example to secdt up your training env.
        '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
        agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12345),
        agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12346),
        agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12347),
        agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12348),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v1', agent_list)
    env.max_steps=500
    print(env.observation_space)
    print(env.action_space)

    # Run the episodes just like OpenAI Gym
    actions = []
    observations = []
    rewards = []
    episode_returns = []
    episode_starts = []
    print("提取专家数据数量: num_traj_e=%d" % num_traj)
    print("随机提取每回合 %d%%的数据" % rand)
    for i_episode in tqdm(range(num_traj)):
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

            if env._agents[0].is_alive:
                obs = featurize(state[0])
                observations.append(obs)
            if env._agents[1].is_alive:
                obs = featurize(state[1])
                observations.append(obs)
            if env._agents[2].is_alive:
                obs = featurize(state[2])
                observations.append(obs)
            if env._agents[3].is_alive:
                obs = featurize(state[3])
                observations.append(obs)

            # 如果第三个智能体死了就不取数据
            # if not env._agents[3].is_alive:
            #     done = True
            #     reward = [0, 0, 0, -1]

            # 随机存取
            # _rand = random.randint(0,100)
            # if _rand < rand or flag:
            #     obs = featurize(state[3])
            #     observations.append(obs)
            #     rewards.append(reward[3])
            #     actions.append(acts[3])

            # flag = False
        print('Episode {} finished'.format(i_episode))
        episode_returns.append(reward[3])
    env.close()

    start = time.time()
    obs = np.array(observations)
    print("去重前shape:", obs.shape)
    print("去重后shape:", np.unique(obs, axis=0).shape)
    end = time.time()
    print("去重所用时间:", end - start)

    numpy_dict = {
        'actions': np.array(actions).reshape(-1, 1),
        'obs': np.array(observations),
        'rewards': np.array(rewards),
        'episode_returns': np.array(episode_returns),
        'episode_starts': np.array(episode_starts)
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)
    print("保存专家数据到: data_path=%s" % data_path)
    np.savez(data_path, **numpy_dict)

if __name__ == '__main__':
    data_path = 'dataset/test_unique.npz'
    print("正在进行专家数据提取, 保存路径为: %s" % data_path)
    _extra(rand=1000, data_path=data_path, num_traj=3)
