import pommerman
from pommerman import agents

import sys
import os

import multiprocessing

import tensorflow as tf

from my_cmd_utils import my_arg_parser
from my_subproc_vec_env import SubprocVecEnv
from my_policies import CustomPolicy

from my_ppo2 import PPO2
from utils import featurize

# TODO：加seed
def make_envs(env_id):
    def _thunk():
        agent_list = [
            # agents.SimpleAgent(),
            # agents.SimpleAgent(),
            # agents.SimpleAgent(),
            # agents.SimpleAgent()
            agents.RandomAgent(),
            agents.SimpleAgent(),
            agents.RandomAgent(),
            agents.SimpleAgent()
        ]
        env = pommerman.make(env_id, agent_list)
        return env
    return _thunk

# def test():
#     print("INIT CONTINURAL PPO2")
#     model = PPO2(CustomPolicy, verbose=1)
#     print("RUN PRETRAIN n_epochs =", 10)
#     from my_dataset import ExpertDataset
#     # dataset = ExpertDataset(expert_path='dataset/test.npz')
#     dataset = ExpertDataset(expert_path='dataset/expert_30/expert_30_0.npz')
#     model.pretrain(dataset=dataset, n_epochs=1)
#     model.pretrain(dataset=dataset, n_epochs=1)
#     # Mutiprocessing
#     config = tf.ConfigProto()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     config.gpu_options.allow_growth = True
#     num_envs = args.num_env or multiprocessing.cpu_count()
#     envs = [make_envs(args.env) for _ in range(num_envs)]
#     env = SubprocVecEnv(envs)
#     model.learn(total_timesteps=100,
#                 seed=args.seed, env=env, using_PGN=True, save_old=True)
#     model.learn(total_timesteps=100,
#                 seed=args.seed, env=env, using_PGN=True, save_old=True)
#     model.save('models/test.zip')

def _pretrain(expert_path,n_epochs):
    if args.load_path:
        print("LOAD MODEL FROM",args.load_path)
        model = PPO2.load(args.load_path)
    else:
        # Init a Continural PPO2 model
        print("INIT CONTINURAL PPO2")
        model = PPO2(CustomPolicy, verbose=1, tensorboard_log=args.log_path)

    print("IN PRETRAIN")
    from my_dataset import ExpertDataset
    # assert args.expert_path is not None

    # load dataset
    print("LOAD DATASET FROM",expert_path)

    print("RUN PRETRAIN n_epochs =", n_epochs)
    dataset = ExpertDataset(expert_path=expert_path)  # traj_limitation 只能取默认-1
    model.pretrain(dataset=dataset, n_epochs=n_epochs)
    del dataset



def train():

    total_timesteps = int(args.num_timesteps)

    # Mutiprocessing
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    num_envs = args.num_env or multiprocessing.cpu_count()
    envs = [make_envs(args.env) for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    if args.load_path:
        print("LOAD A MODEL FOR TRAIN FROM", args.load_path)
        model = PPO2.load(args.load_path)

    print("START TO TRAIN")
    print("USING ENVIRONMEN", args.env)
    print("TOTAL_TIMESTEPS =",total_timesteps)
    print("IS USING PGN",args.using_PGN)
    print("IS SAVE OLD PARAMS",args.save_old)

    model.learn(total_timesteps=total_timesteps,
                seed=args.seed, env=env, using_PGN=args.using_PGN, save_old=args.save_old)

    if args.save_path:
        print("SAVE LEARNED MODEL", args.save_path)
        model.save(save_path=args.save_path)

    env.close()


def play(train_idx):
    if not args.load_path:
        print('PLAY NEED --load_path')
        raise ValueError

    model = PPO2.load(args.load_path)
    # print(pommerman.REGISTRY)
    print('LOAD MODEL FROM', args.load_path)
    agent_list = [
        # agents.SimpleNoBombAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12346),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12345),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12344),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12343),
    ]
    env = pommerman.make('PommeRadioCompetition-v2',agent_list)

    # Index of test agent
    # env.set_training_agent(train_idx)

    for episode in range(100):
        obs = env.reset()
        done = False
        while not done:
            feature = featurize(obs[train_idx])
            action, _states = model.predict(feature)
            print(action)
            all_actions = env.act(obs)
            all_actions[train_idx] = int(action)
            obs, rewards, done, info = env.step(all_actions)
            env.render()
            if not env._agents[train_idx].is_alive:
                done = True
def _evaluate(train_idx,n_episode):
    if not args.load_path:
        print('PLAY NEED --load_path')
        raise ValueError
    model = PPO2.load(args.load_path)

    # Index of test agent
    print(pommerman.REGISTRY)
    print('LOAD MODEL FROM', args.load_path)
    agent_list = [
        # agents.SimpleNoBombAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12346),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12345),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12344),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12343),
    ]
    env = pommerman.make('PommeRadioCompetition-v2', agent_list)
    win = 0
    tie = 0
    loss = 0
    for episode in range(n_episode):
        obs = env.reset()
        done = False
        while not done:
            feature = featurize(obs[train_idx])
            action, _states = model.predict(feature)
            all_actions = env.act(obs)
            # print(train_idx)
            # print(all_actions)
            # print(int(action))
            all_actions[train_idx] = int(action)
            obs, rewards, done, info = env.step(all_actions)
            # env.render()
        if rewards[train_idx] == 1:
            win += 1
        elif rewards[train_idx] == -1:
            loss += 1
        else:
            tie += 1
        if episode % 100 == 0:
            print("win / tie / loss")
            print(" %d  /  %d  /  %d " % (win,tie,loss))
    env.close()


if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    # Test
    # test()

    # Pretrain
    if args.pre_train:
        _pretrain('dataset/v0/',10000)

    # Train
    if args.train:
        train()

    # Test
    if args.play:
        play(0)
    #
    # Evaluate
    if args.evaluate:
        _evaluate(0,10000)