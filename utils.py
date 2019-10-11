import numpy as np
from gym import spaces


def featurize(obs):
    maps = []
    # TODO: 爆炸范围内的位置手动置 1，还要考虑地图空白部分。还有另外两个11*11的特征
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

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
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)
    """两个敌人的位置"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            maps.append(board == e.value)
    """训练智能体的位置"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*18


def get_feature_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 18))


def get_action_space():
    return spaces.Discrete(6)
