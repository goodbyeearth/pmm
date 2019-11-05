import gym

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
from stable_baselines import GAIL


agent_list = [
        # agents.RandomAgent(),
        agents.SuperAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]
env = pommerman.make('PommeRadioCompetition-v2', agent_list)

expert_path = 'dataset/test/'
from my_dataset import ExpertDataset

# load dataset
print("Load dataset from", expert_path)
dataset = ExpertDataset(expert_path=expert_path)  # traj_limitation 只能取默认-1

model = GAIL(CustomPolicy, dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=10, env=env)
model.save('models/test.zip')

del model  # remove to demonstrate saving and loading
