import numpy as np
from utils import featurize
import random
from tqdm import tqdm
import pommerman
from pommerman import agents
import time


def _extra(data_path, num_traj):
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
        agents.SimpleAgent(),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12345),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12346),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12347),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12348),
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    # env.max_steps=500
    print(env.observation_space)
    print(env.action_space)

    # Run the episodes just like OpenAI Gym
    actions = []
    observations = []
    rewards = []
    episode_returns = []
    print("NUM OF EXTRACTED EXPERT DATA", num_traj)
    for i_episode in tqdm(range(num_traj)):
        start = time.time()
        state = env.reset()
        tmp_ob = [[],[],[],[]]
        tmp_act = [[],[],[],[]]
        tmp_r = [[],[],[],[]]
        done = False
        while not done:
            # env.render()
            obs = [[],[],[],[]]
            acts = env.act(state)
            state, reward, done, info = env.step(acts)
            for _i in range(4):
                obs[_i].append(featurize(state[_i]))
                # print(np.array(obs[_i]).shape)
                tmp_ob[_i].append(obs[_i])
                tmp_r[_i].append(reward[_i])
                tmp_act[_i].append(acts[_i])
                # print(tmp_act)

            # flag = False
        print('Episode {} finished'.format(i_episode))
        if _i in range(4):
            if reward[_i] == 1:
                observations.extend(tmp_ob[_i])
                actions.extend(tmp_act[_i])
                rewards.extend(tmp_r[_i])
                episode_returns.append(reward[_i])
        end = time.time()
        print("TIME:", end - start)
    env.close()

    # start = time.time()
    # obs = np.array(observations)
    # print("BEFORE UNIQUE SHAPE", obs.shape)
    # print("AFTER UNIQUE SHAPE", np.unique(obs, axis=0).shape)
    # end = time.time()
    # print("UNIQUE TIME", end - start)

    numpy_dict = {
        'actions': np.array(actions).reshape(-1, 1),
        'obs': np.array(observations).reshape(-1,11,11,18),
        'rewards': np.array(rewards),
        'episode_returns': np.array(episode_returns),
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)
    print("SAVE DATA data_path=%s" % data_path)
    np.savez(data_path, **numpy_dict)


if __name__ == '__main__':
    data_path = 'dataset/all_simple_1w.npz'
    print("EXTRA DATA SAVE IN %s" % data_path)
    _extra(data_path=data_path, num_traj=2000)
