import numpy as np
from gym import spaces

def featurize(obs, train_agent_idx):
    obs = obs[train_agent_idx]    # 取训练智能体的 observation
    maps = []

    # TODO: 爆炸范围内的位置手动置 1，还要考虑地图空白部分。还有另外两个11*11的特征
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    board = obs['board']
    """棋盘物体 one hot"""
    for i in range(9):   # [0, 1, ..., 8]
        maps.append(board == i)
    """训练智能体的位置"""
    player = train_agent_idx + 10
    maps.append(board == player)
    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))
    """队友及敌人的位置"""
    maps.append(board == obs['teammate'].value)
    for e in obs['enemies']:
        if not e.value == 9:   # AgentDummy
            maps.append(board == e.value)

    return np.stack(maps, axis=2)


def get_feature_shape():
    return (11, 11, 18)


def get_action_space():
    return spaces.MultiDiscrete([6, 8, 8])