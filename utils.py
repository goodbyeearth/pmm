import numpy as np
from gym import spaces


def _featurize0(obs):
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


def featurize(obs):
    maps = []
    board = obs['board']
    """棋盘物体 one hot"""
    for i in range(9):  # [0, 1, ..., 8]
        maps.append(board == i)

    '''爆炸one-hot'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']

    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    for i in range(2, 13):
        maps.append(bomb_life == i)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

    """标量映射为11*11的矩阵"""
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    # maps.append(np.full(board.shape, obs['step_count']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # alive = obs['alive']
    # fog = board
    # fog = np.where(fog == 5, 5, 0)

    """一个队友的位置 one-hot """
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)

    # if teammate_idx in alive:
    #     t_flag = 1
    # else:
    #     t_flag = 0

    """两个敌人的位置 one-hot"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            maps.append(board == e.value)

    """fog的情况"""
    # e_flag = 0
    # for e in enemies_idx:
    #     if e in alive:
    #         e_flag += 1
    # if t_flag == 1:
    #     fog = np.where(fog == 5, 5 + e_flag, 0)
    # else:
    #     fog = np.where(fog == 5, 3 + e_flag, 0)
    # for i in range(4, 8):
    #     maps.append(fog == i)

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break

    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*31
    # return np.stack(maps, axis=2),bomb_life


def get_feature_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 31))


def get_action_space():
    return spaces.Discrete(6)


def get_bomb_life(obs):
    board = obs['board']
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']
    # 统一炸弹时间
    for x in range(11):
        for y in range(11):
            if bomb_blast_strength[(x, y)] > 0:
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x + i, y)
                    if x + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x - i, y)
                    if x - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y + i)
                    if y + i > 10:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]
                for i in range(1, int(bomb_blast_strength[(x, y)])):
                    pos = (x, y - i)
                    if y - i < 0:
                        break
                    if board[pos] == 1:
                        break
                    if board[pos] == 2:
                        bomb_life[pos] = bomb_life[(x, y)]
                        break
                    # if a bomb
                    if board[pos] == 3:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                        else:
                            bomb_life[(x, y)] = bomb_life[pos]
                    elif bomb_life[pos] != 0:
                        if bomb_life[(x, y)] < bomb_life[pos]:
                            bomb_life[pos] = bomb_life[(x, y)]
                    else:
                        bomb_life[pos] = bomb_life[(x, y)]

    bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
    flame_life = np.where(flame_life == 0, 15, flame_life)
    flame_life = np.where(flame_life == 1, 15, flame_life)
    bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
    return bomb_life


def judge_bomb(obs):
    bomb_life = get_bomb_life(obs)
    if (bomb_life > 0).any():
        return True  # 有爆炸威胁
    else:
        return False  # 无爆炸威胁


def judge_bomb_4(obs):
    pos = next_pos(obs['position'])
    bomb_life = get_bomb_life(obs)
    for p in pos:
        if bomb_life[p] > 0 or 11 in obs['board'] or 13 in obs['board']:
            return True
    return False


def next_pos(pos):
    x, y = pos
    up = (np.clip(x - 1, 0, 10), y)
    down = (np.clip(x + 1, 0, 10), y)
    left = (x, np.clip(y - 1, 0, 10))
    right = (x, np.clip(y + 1, 0, 10))
    return [pos, up, down, left, right]

# def _featurize(obs):
#     maps = []
#
#     """棋盘物体 one hot"""
#     board = obs['board']
#     for i in range(9):  # [0, 1, ..., 8]
#         maps.append(board == i)
#
#     '''爆炸one-hot'''
#     bomb_life = obs['bomb_life']
#     bomb_blast_strength = obs['bomb_blast_strength']
#     flame_life = obs['flame_life']
#     # 统一炸弹时间
#     for x in range(11):
#         for y in range(11):
#             if bomb_blast_strength[(x, y)] > 0 :
#                 for i in range(1, int(bomb_blast_strength[(x, y)])):
#                     pos = (x + i, y)
#                     if x + i > 10:
#                         break
#                     if board[pos] == 1:
#                         break
#                     if board[pos] == 2:
#                         bomb_life[pos] = bomb_life[(x,y)]
#                         break
#                     # if a bomb
#                     if board[pos] == 3:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                         else:
#                             bomb_life[(x, y)] = bomb_life[pos]
#                     elif bomb_life[pos] != 0:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                     else:
#                         bomb_life[pos] = bomb_life[(x, y)]
#                 for i in range(1, int(bomb_blast_strength[(x, y)])):
#                     pos = (x - i, y)
#                     if x - i < 0:
#                         break
#                     if board[pos] == 1:
#                         break
#                     if board[pos] == 2:
#                         bomb_life[pos] = bomb_life[(x,y)]
#                         break
#                     # if a bomb
#                     if board[pos] == 3:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                         else:
#                             bomb_life[(x, y)] = bomb_life[pos]
#                     elif bomb_life[pos] != 0:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                     else:
#                         bomb_life[pos] = bomb_life[(x, y)]
#                 for i in range(1, int(bomb_blast_strength[(x, y)])):
#                     pos = (x, y + i)
#                     if y + i > 10:
#                         break
#                     if board[pos] == 1:
#                         break
#                     if board[pos] == 2:
#                         bomb_life[pos] = bomb_life[(x,y)]
#                         break
#                     # if a bomb
#                     if board[pos] == 3:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                         else:
#                             bomb_life[(x, y)] = bomb_life[pos]
#                     elif bomb_life[pos] != 0:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                     else:
#                         bomb_life[pos] = bomb_life[(x, y)]
#                 for i in range(1, int(bomb_blast_strength[(x, y)])):
#                     pos = (x, y - i)
#                     if y - i < 0:
#                         break
#                     if board[pos] == 1:
#                         break
#                     if board[pos] == 2:
#                         bomb_life[pos] = bomb_life[(x,y)]
#                         break
#                     # if a bomb
#                     if board[pos] == 3:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                         else:
#                             bomb_life[(x, y)] = bomb_life[pos]
#                     elif bomb_life[pos] != 0:
#                         if bomb_life[(x, y)] < bomb_life[pos]:
#                             bomb_life[pos] = bomb_life[(x, y)]
#                     else:
#                         bomb_life[pos] = bomb_life[(x, y)]
#
#
#
#     bomb_life = np.where(bomb_life > 0, bomb_life + 3, bomb_life)
#     flame_life = np.where(flame_life == 0, 15, flame_life)
#     bomb_life = np.where(flame_life != 15, flame_life, bomb_life)
#     for i in range(1, 13):
#         maps.append(bomb_life == i)
#     # for i in range(1, 9):
#     #     maps.append(bomb == i)
#     # for i in range(1,3):
#     #     maps.append(flame_life == i)
#     '''将bomb direction编码为one-hot'''
#     bomb_moving_direction = obs['bomb_moving_direction']
#     for i in range(1, 5):
#         maps.append(bomb_moving_direction == i)
#
#     """标量映射为11*11的矩阵"""
#     maps.append(np.full(board.shape, obs['ammo']))
#     maps.append(np.full(board.shape, obs['blast_strength']))
#     maps.append(np.full(board.shape, obs['step_count']))
#     maps.append(np.full(board.shape, obs['can_kick']))
#
#     """一个队友的位置 one-hot """
#     teammate_idx = obs['teammate'].value
#     if not teammate_idx == 9:  # AgentDummy
#         maps.append(board == teammate_idx)
#     """两个敌人的位置 one-hot"""
#     enemies_idx = []
#     for e in obs['enemies']:
#         if not e.value == 9:  # AgentDummy
#             enemies_idx.append(e.value)
#             maps.append(board == e.value)
#     """训练智能体的位置 one-hot"""
#     for idx in [10, 11, 12, 13]:
#         if idx not in enemies_idx + [teammate_idx]:
#             train_agent_idx = idx
#             break
#     maps.append(board == train_agent_idx)
#     """Map to position 10's obs"""
#     maps = np.array(maps)
#     # print(maps.shape)
#     if train_agent_idx == 11:
#         maps = turn_180(maps)
#         # print(maps.shape)
#         maps = turn_image(maps)
#         # print(maps.shape)
#     elif train_agent_idx == 12:
#         maps = turn_180(maps)
#     elif train_agent_idx == 13:
#         maps = turn_image(maps)
#     # print(maps.shape)
#
#
#     return np.stack(maps, axis=2)  # 11*11*33
#
# def turn_image(arr):
#     return arr.T[::-1].transpose()
#
# def turn_180(arr):
#     shape = arr.shape
#     return arr.reshape(arr.size)[::-1].reshape(shape)
