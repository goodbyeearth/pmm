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

from my_ppo2 import PPO2
# from stable_baselines import PPO2
from generate_data import generate_expert_data, merge_data

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

def _pretrain(expert_path,n_epochs):
    print("IN PRETRAIN")
    from my_dataset import ExpertDataset
    # assert args.expert_path is not None

    # load dataset
    print("LOAD DATASET FROM",expert_path)
    dataset = ExpertDataset(expert_path=expert_path)  # traj_limitation 只能取默认-1

    # Init a Continural PPO2 model
    print("INIT CONTINURAL PPO2")
    model = PPO2(CustomPolicy, verbose=1, tensorboard_log=args.log_path)

    print("RUN PRETRAIN n_epochs =", n_epochs)
    model.pretrain(dataset=dataset, n_epochs=n_epochs)

    print("Save prtrain model", args.save_path)
    model.save('./models/model0.zip')

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

    from utils import featurize

    model = PPO2.load(args.load_path)
    env_fn = make_envs(args.env)
    env = env_fn()

    # Index of test agent
    env.set_training_agent(train_idx)

    def get_all_actions():
        feature = featurize(obs[train_idx])
        action, _states = model.predict(feature)
        action = (action, 0, 0)
        some_actions = env.act(obs)  # 不包含我的 agent
        # 如果其他智能体动作不是元组（只有单一动作），改成元组
        for agent_idx in range(3):
            if not isinstance(some_actions[agent_idx], tuple):
                some_actions[agent_idx] = (some_actions[agent_idx], 0, 0)
        some_actions.insert(train_idx, action)  # 把我的 agent 的动作也加进来

        return some_actions

    for episode in range(1):
        obs = env.reset()
        for i in range(1000):
            all_actions = get_all_actions()
            obs, rewards, done, info = env.step(all_actions)
            if done:
                break
            env.render()
    env.close()

def _evaluate(train_idx,n_episode):
    if not args.load_path:
        print('PLAY NEED --load_path')
        raise ValueError

    from utils import featurize

    model = PPO2.load(args.load_path)
    env_fn = make_envs(args.env)
    env = env_fn()

    # Index of test agent
    env.set_training_agent(train_idx)

    def get_all_actions():
        feature = featurize(obs[train_idx])
        action, _states = model.predict(feature)
        action = (action, 0, 0)
        some_actions = env.act(obs)  # 不包含我的 agent
        # 如果其他智能体动作不是元组（只有单一动作），改成元组
        for agent_idx in range(3):
            if not isinstance(some_actions[agent_idx], tuple):
                some_actions[agent_idx] = (some_actions[agent_idx], 0, 0)
        some_actions.insert(train_idx, action)  # 把我的 agent 的动作也加进来

        return some_actions
    win = 0
    tie = 0
    loss = 0
    for episode in range(n_episode):
        obs = env.reset()
        done = True
        while done:
            all_actions = get_all_actions()
            obs, rewards, done, info = env.step(all_actions)
            if rewards[train_idx] == 1:
                win += 1
            elif rewards[train_idx] == -1:
                loss += 1
            else:
                tie += 1
            env.render()
        if episode % 1000 == 0:
            print("win/tie/loss: {}/{}/{}" % win,tie,loss)
    env.close()

if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    # Pretrain
    if args.pre_train:
        _pretrain('dataset/expert_30/',5)

    # Train
    if args.train:
        train()

    # Test
    if args.play:
        play(3)

    # Evaluate
    if args.evaluate:
        _evaluate(3,10000)