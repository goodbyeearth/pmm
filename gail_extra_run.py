'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents
import numpy as np
from utils import featurize
import gym

def main():
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
    ]

    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeTeamCompetition-v1', agent_list)
    print(env.observation_space)
    print(env.action_space)


    # Run the episodes just like OpenAI Gym
    actions=[]
    observations=[]
    rewards=[]
    episode_returns=[]
    episode_starts=[]

    for i_episode in range(10):
        state = env.reset()
        flag = True
        done = False
        while not done:
            if flag:
                episode_starts.append(True)
                flag = False
            else:
                episode_starts.append(False)
            # env.render()
            acts = env.act(state)
            state, reward, done, info = env.step(acts)
            if not env._agents[3].is_alive:
                done = True
                reward = [0,0,0,-1]

            obs = featurize(state[3])
            observations.append(obs)
            rewards.append(reward[3])
            actions.append(acts[3])
        print('Episode {} finished'.format(i_episode))
        episode_returns.append(reward[3])
    env.close()

    numpy_dict = {
        'actions': np.array(actions).reshape(-1,1),
        'obs': np.array(observations),
        'rewards': np.array(rewards),
        'episode_returns': np.array(episode_returns),
        'episode_starts': np.array(episode_starts)
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    np.savez('dataset/pmm', **numpy_dict)

if __name__ == '__main__':
    main()
