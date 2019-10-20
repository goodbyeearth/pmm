import numpy as np
from gym import spaces


def featurize(obs):
    maps = []
    # TODO: 爆炸范围内的位置手动置 1，还要考虑地图空白部分。还有另外两个11*11的特征
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    board = obs['board']
    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
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


def _featurize(obs):
    maps = []
    '''统一炸弹时间 并one-hot'''
    bomb = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    for x in range(11):
        for y in range(11):
            if bomb[(x, y)] != 0:
                tmp_life = bomb[(x, y)]
                tmp_strength = bomb_blast_strength[(x, y)]
                for _i in range(int(tmp_strength)):
                    _ = _i + 1
                    x3 = x - _ if x >= _ else x
                    x4 = x + _ if x + _ < 11 else 10
                    y1 = y - _ if y >= _ else y
                    y2 = y + _ if y + _ < 11 else 10
                    # print(x3,x4,y1,y2)
                    if bomb[(x3, y)] > tmp_life + _:
                        bomb[(x3, y)] = tmp_life + _
                    if bomb[(x4, y)] > tmp_life + _:
                        bomb[(x4, y)] = tmp_life + _
                    if bomb[(x, y1)] > tmp_life + _:
                        bomb[(x, y1)] = tmp_life + _
                    if bomb[(x, y2)] > tmp_life + _:
                        bomb[(x, y2)] = tmp_life + _
    for x in range(11):
        for y in range(11):
            if bomb[(x, y)] != 0:
                # print(x,y)
                tmp_life = bomb[(x, y)]
                tmp_strength = bomb_blast_strength[(x, y)]
                # print(tmp_life)
                # print(tmp_strength)
                for _i in range(int(tmp_strength)):
                    _ = _i+1
                    x3 = x - _ if x >= _ else x
                    x4 = x + _ if x+_ < 11 else 10
                    y1 = y - _ if y >= _ else y
                    y2 = y + _ if y+_ < 11 else 10
                    # print(x3,x4,y1,y2)
                    if bomb[(x3, y)] == 0 or bomb[(x3, y)] > tmp_life + _:
                        bomb[(x3, y)] = tmp_life + _
                    if bomb[(x4, y)] == 0 or bomb[(x4, y)] > tmp_life + _:
                        bomb[(x4, y)] = tmp_life + _
                    if bomb[(x, y1)] == 0 or bomb[(x, y1)] > tmp_life + _:
                        bomb[(x, y1)] = tmp_life + _
                    if bomb[(x, y2)] == 0 or bomb[(x, y2)] > tmp_life + _:
                        bomb[(x, y2)] = tmp_life + _
    for x in range(11):
        for y in range(11):
            if bomb[(x,y)] > 0:
                bomb[(x,y)] += 3
    for i in range(13):
        maps.append(bomb == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(3):
        _i = i+1
        maps.append(bomb_moving_direction == _i)

    board = obs['board']
    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    """一个队友的位置 one-hot """
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)
    """两个敌人的位置 one-hot"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            maps.append(board == e.value)
    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*32
