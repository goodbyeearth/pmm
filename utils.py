import numpy as np
from gym import spaces


def featurize(obs):
    maps = []
    maps.append(obs['bomb_blast_strength']/5)
    # bomb_life = 11 - obs['bomb_life']
    # bomb_life =
    # maps.append(obs['bomb_life']/5)
    # maps.append(obs['flame_life']/3)
    for i in range(1, 4):
        maps.append(obs['flame_life'] == i)
    for i in range(1, 11):
        maps.append(obs['bomb_life'] == i)

    board = obs['board']
    """棋盘物体 one hot"""
    for i in range(9):   # [0, 1, ..., 8]
        maps.append(board == i)
    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))
    """一个队友的位置"""
    teammate_idx = obs['teammate'].value
    maps.append(board == teammate_idx)
    """两个敌人的位置"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            maps.append(board == e.value)
    """训练智能体的位置"""
    train_agent_idx = None
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    assert train_agent_idx is not None
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)


def get_feature_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 30))


def get_action_space():
    return spaces.Discrete(6)
