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
from utils import *
import numpy as np
import time


# TODO：加seed
def make_envs(env_id):
    def _thunk():
        agent_list = [
            agents.SuperAgent(),
            agents.StopAgent(),
            agents.SuicideAgent(),
            agents.StopAgent()
        ]
        env = pommerman.make(env_id, agent_list)
        return env

    return _thunk


def _pretrain():
    expert_path = 'dataset/hako1/agent_0/'

    if args.load_path:
        print("Load model from", args.load_path)
        model = PPO2.load(args.load_path, using_PGN=args.using_PGN)
    else:
        # Init a Continural PPO2 model
        print("Init a pgn PPO2")
        model = PPO2(CustomPolicy, verbose=1, tensorboard_log=args.log_path)

    print("In pretrain")
    print()

    from my_dataset import ExpertDataset
    # assert args.expert_path is not None
    # load dataset
    print("Load dataset from", expert_path)
    print()
    print("Run pretrain n_epochs =", int(args.num_timesteps))
    print()
    dataset = ExpertDataset(expert_path=expert_path)  # traj_limitation 只能取默认-1
    model.pretrain(dataset=dataset, n_epochs=int(args.num_timesteps), save_path=args.save_path)
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
        print()
        model = PPO2.load(args.load_path, using_PGN=args.using_PGN, tensorboard_log=args.log_path)
    else:
        print("INIT CONTINURAL PPO2")
        print()
        model = PPO2(CustomPolicy, verbose=1, tensorboard_log=args.log_path)

    print("START TO TRAIN")
    print("USING ENVIRONMEN", args.env)
    print("TOTAL_TIMESTEPS =", total_timesteps)
    print("IS USING PGN", args.using_PGN)
    print()

    model.learn(total_timesteps=total_timesteps,
                seed=args.seed, env=env, save_path=args.save_path, save_interval=args.save_interval)

    # if args.save_path:
    #     print("SAVE LEARNED MODEL", args.save_path)
    #     model.save(save_path=args.save_path)

    env.close()


def modify_act(obs, act):
    from pommerman.agents.prune import get_filtered_actions
    import random
    valid_actions = get_filtered_actions(obs.copy())
    if act not in valid_actions:
        act = random.sample(valid_actions, 1)
    return act


def play0():
    model0_path = 'models/hako/agent_0/hako_e10.zip'
    model0 = PPO2.load(model0_path)

    model2_path = 'models/hako/agent_0/hako_e10.zip'
    model2 = PPO2.load(model2_path)

    agent_list = [
        # agents.SimpleNoBombAgent(),
        agents.SuicideAgent(),
        agents.SuperAgent(),
        agents.SuicideAgent(),
        agents.SuperAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12347),
    ]
    env = pommerman.make(args.env, agent_list)
    env._max_steps = 500
    print('Load model0 from', model0_path)
    print('Load model2 from', model2_path)
    print("play0 --> train_idx (0,2)")

    for episode in range(100):
        obs = env.reset()
        done = False
        while not done:
            all_actions = env.act(obs)

            feature0 = featurize(obs[0])  # model0
            action0, _states = model0.predict(feature0)
            action0 = int(action0)
            all_actions[0] = action0

            feature2 = featurize(obs[2])  # model2
            action2, _states = model2.predict(feature2)
            action2 = int(action2)
            all_actions[2] = action2

            obs, rewards, done, info = env.step(all_actions)
            env.render()
            # if not env._agents[0].is_alive:
            #     done = True
        print(info)


def play1():
    model0 = PPO2.load(args.load_path)

    agent_list = [
        # agents.SimpleNoBombAgent(),
        agents.SimpleAgent(),
        agents.StopAgent(),
        agents.SuicideAgent(),
        agents.StopAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12347),
    ]
    env = pommerman.make(args.env, agent_list)
    # env._max_steps = 500
    print('Load model0 from', args.load_path)
    print("play1 --> train_idx 0")
    for episode in range(100):
        obs = env.reset()
        done = False
        flag = True
        while not done:
            all_actions = env.act(obs)
            if judge_enemy(obs[0]):
                feature0 = featurize(obs[0])  # model0
                action0, _states = model0.predict(feature0)
                all_actions[0] = int(action0)
            #     if flag:
            #         print(">>>> model:", action0)
            #         flag = False
            #     else:
            #         flag = True
            #         print("<<<< model:", action0)
            # else:
            #     print("super")

            # feature2 = featurize(obs[2])  # model2
            # action2, _states = model2.predict(feature2)
            # all_actions[2] = int(action2)

            obs, rewards, done, info = env.step(all_actions)
            env.render()
            if not env._agents[0].is_alive:
                done = True
                print("death")
        print(rewards)
        print(info)


def _evaluate(n_episode=10000):
    from pommerman import constants
    model0_path = 'models/simple/agent_0/simple_e40.zip'
    model2_path = 'models/simple/agent_2/simple_e40.zip'
    model0 = PPO2.load(model0_path)
    model2 = PPO2.load(model2_path)

    agent_list = [
        # agents.SimpleNoBombAgent(),
        agents.SuicideAgent(),
        agents.SuperAgent(),
        agents.SuicideAgent(),
        agents.SuperAgent(),
        # agents.PlayerAgent(agent_control="arrows"),
        # agents.DockerAgent("multiagentlearning/hakozakijunctions", port=12343),
    ]
    env = pommerman.make(args.env, agent_list)
    print('Load model0 from', model0_path)
    print('Load model2 from', model2_path)
    print("evaluate --> train_idx (0,2)")
    print("my agents vs simple")
    win = 0
    tie = 0
    loss = 0
    for episode in range(n_episode):
        obs = env.reset()
        done = False
        start = time.time()
        while not done:
            all_actions = env.act(obs)  # get all actions

            feature0 = featurize(obs[0])  # model0
            action0, _states = model0.predict(feature0)
            action0 = int(action0)
            from pommerman.agents.prune import get_filtered_actions
            import random
            valid_actions_0 = get_filtered_actions(obs[0])
            if action0 not in valid_actions_0:
                action0 = random.sample(valid_actions_0, 1)
            all_actions[0] = action0

            feature2 = featurize(obs[2])  # model2
            action2, _states = model2.predict(feature2)
            action2 = int(action2)
            valid_actions_2 = get_filtered_actions(obs[2])
            if action2 not in valid_actions_2:
                action2 = random.sample(valid_actions_2, 1)
            all_actions[2] = action2

            valid_actions_1 = get_filtered_actions(obs[1])
            if all_actions[1] not in valid_actions_1:
                all_actions[1] = random.sample(valid_actions_1, 1)

            valid_actions_3 = get_filtered_actions(obs[3])
            if all_actions[3] not in valid_actions_3:
                all_actions[3] = random.sample(valid_actions_3, 1)

            obs, rewards, done, info = env.step(all_actions)
            # env.render()
        if info['result'] == constants.Result.Tie:
            tie += 1
        elif info['winners'] == [0, 2]:
            win += 1
        else:
            loss += 1
        end = time.time()
        if (episode + 1) % 10 == 0:
            print("win / tie / loss")
            print(
                " %d  /  %d  /  %d  win rate: %f  use time: %f" % (win, tie, loss, (win / (episode + 1)), end - start))
    env.close()


if __name__ == '__main__':
    arg_parser = my_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    # Pretrain
    if args.pre_train:
        _pretrain()

    # Pretrain_test
    # if args.pre_train:
    #     _pretrain('dataset/test/',10)

    # Train
    if args.train:
        train()

    # Test
    if args.play == 0:
        play0()
    elif args.play == 1:
        play1()
    #
    # Evaluate
    if args.evaluate:
        _evaluate()

    # def obs_map(j,arr):
    #     if j == 0:
    #         # print(flip_right(arr))
    #         return arr
    #     if j == 1:
    #         # print(flip_right(arr))
    #         return flip_right(arr)
    #     if j == 2:
    #         # print(flip_right(arr))
    #         return flip180(arr)
    #     if j == 3:
    #         # print(flip_right(arr))
    #         return flip_left(arr)
    # def act_map(j,act):
    #     if j == 0:
    #         return act
    #     if j == 1:
    #         return turn_right(act)
    #     if j == 2:
    #         return turn_180(act)
    #     if j == 3:
    #         return turn_left(act)
    # def pos_map(j, pos):
    #     if j == 0:
    #         return pos
    #     if j == 1:
    #         return pos_right(pos)
    #     if j == 2:
    #         pos = pos_right(pos)
    #         return pos_right(pos)
    #     if j == 3:
    #         return pos_left(pos)
    # def flip180(arr):
    #     new_arr = arr.reshape(arr.size)
    #     new_arr = new_arr[::-1]
    #     new_arr = new_arr.reshape(arr.shape)
    #     # print("180")
    #     return new_arr
    # def flip_left(arr):
    #     new_arr = np.transpose(arr)
    #     new_arr = new_arr[::-1]
    #     # print("left")
    #     return new_arr
    # def flip_right(arr):
    #     new_arr = arr.reshape(arr.size)
    #     new_arr = new_arr[::-1]
    #     new_arr = new_arr.reshape(arr.shape)
    #     new_arr = np.transpose(new_arr)[::-1]
    #     # print("right")
    #     return new_arr
    # def turn_right(act):
    #     map = [0,4,3,1,2,5]
    #     return map[int(act)]
    # def turn_left(act):
    #     map = [0,3,4,2,1,5]
    #     return map[int(act)]
    # def turn_180(act):
    #     map = [0,2,1,4,3,5]
    #     return map[int(act)]
    # def pos_right(pos):
    #     x,y = pos
    #     x = x-5
    #     y = y-5
    #     x1 = y
    #     y1 = -x
    #     x = x1+5
    #     y = y1+5
    #     return (x,y)
    # def pos_left(pos):
    #     x,y = pos
    #     x = x-5
    #     y = y-5
    #     x1 = -y
    #     y1 = x
    #     x = x1+5
    #     y = y1+5
    #     return (x,y)
    # def state_map(obs):
    #     # print(obs)
    #     n = obs['board'][obs['position']]
    #     if n not in [10,11,12,13]:
    #         return 0,obs
    #     n -= 10
    #     # print(n)
    #     obs['board'] = obs_map(n, obs['board'])
    #     obs['bomb_blast_strength'] = obs_map(n, obs['bomb_blast_strength'])
    #     obs['bomb_life'] = obs_map(n, obs['bomb_life'])
    #     # print(obs_map(n, obs['bomb_moving_direction']))
    #     obs['bomb_moving_direction'] = obs_map(n, obs['bomb_moving_direction'])
    #     # print(obs['bomb_moving_direction'])
    #     for x in range(11):
    #         for y in range(11):
    #             obs['bomb_moving_direction'][(x, y)] = act_map(n,obs['bomb_moving_direction'][(x, y)])
    #     obs['flame_life'] = obs_map(n, obs['flame_life'])
    #     obs['position'] = pos_map(n, obs['position'])
    #     return n,obs
    # def act_back(n,act):
    #     if n == 0:
    #         return act
    #     if n == 1:
    #         return turn_left(act)
    #     if n == 2:
    #         return turn_180(act)
    #     if n == 3:
    #         return turn_right(act)

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
