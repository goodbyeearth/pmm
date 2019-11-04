import numpy as np
from gym import spaces

def get_feature_space():
    return spaces.Box(low=0, high=1, shape=(11, 11, 30))

def get_action_space():
    return spaces.Discrete(6)

#11*11*20
def featurize_bak(obs):
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

    maps.append(bomb_life)

    '''将bomb direction编码为one-hot'''
    bomb_moving_direction = obs['bomb_moving_direction']
    for i in range(1, 5):
        maps.append(bomb_moving_direction == i)

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
            # maps.append(board == e.value)

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break

    eboard = board
    for ex in range(11):
        for ey in range(11):
            if eboard[(ex, ey)] == ((train_agent_idx - 10 + 1) % 4) + 10 or eboard[(ex, ey)] == (
                    (train_agent_idx - 10 + 3) % 4) + 10:
                eboard[(ex, ey)] = True
            else:
                eboard[(ex, ey)] = False
    maps.append(eboard)

    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*20

# 11*11*30
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

    """一个队友的位置 one-hot """
    teammate_idx = obs['teammate'].value
    if not teammate_idx == 9:  # AgentDummy
        maps.append(board == teammate_idx)

    """两个敌人的位置 one-hot"""
    enemies_idx = []
    for e in obs['enemies']:
        if not e.value == 9:  # AgentDummy
            enemies_idx.append(e.value)
            # maps.append(board == e.value)

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

    eboard = board
    for ex in range(11):
        for ey in range(11):
            if eboard[(ex, ey)] == ((train_agent_idx - 10 + 1) % 4) + 10 or eboard[(ex, ey)] == (
                    (train_agent_idx - 10 + 3) % 4) + 10:
                eboard[(ex, ey)] = True
            else:
                eboard[(ex, ey)] = False
    maps.append(eboard)

    return np.stack(maps, axis=2)  # 11*11*31
    # return np.stack(maps, axis=2),bomb_lifes

# 11*11*9
def featurize_bak(obs):  # no one-hot
    maps = []
    board = obs['board']
    maps.append(board)

    '''爆炸'''
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
    maps.append(bomb_life)

    '''bomb direction'''
    bomb_moving_direction = obs['bomb_moving_direction']
    maps.append(bomb_moving_direction)

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
            # maps.append(board == e.value

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    eboard = board
    for ex in range(11):
        for ey in range(11):
            if eboard[(ex, ey)] == ((train_agent_idx - 10 + 1) % 4) + 10 or eboard[(ex, ey)] == (
                    (train_agent_idx - 10 + 3) % 4) + 10:
                eboard[(ex, ey)] = True
            else:
                eboard[(ex, ey)] = False
    maps.append(eboard)
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*9

# 11*11*16
def featurize_bak(obs):  # no one-hot
    maps = []
    board = obs['board']
    maps.append(board)

    powerup = board
    ragids = board
    dangers = board

    for ex in range(11):
        for ey in range(11):
            if powerup[ex, ey] not in [6, 7, 8]:
                powerup[ex, ey] = 0
    maps.append(powerup)

    for ex in range(11):
        for ey in range(11):
            if ragids[ex, ey] not in [1, 2]:
                ragids[ex, ey] = 0
    maps.append(ragids)

    for ex in range(11):
        for ey in range(11):
            if dangers[ex, ey] not in [3, 4]:
                dangers[ex, ey] = 0
    maps.append(dangers)

    maps.append(board == 5)

    '''爆炸'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']

    maps.append(bomb_life)
    maps.append(bomb_blast_strength)
    maps.append(flame_life)

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
    maps.append(bomb_life)

    '''bomb direction'''
    bomb_moving_direction = obs['bomb_moving_direction']
    maps.append(bomb_moving_direction)

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
            # maps.append(board == e.value

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    eboard = board
    for ex in range(11):
        for ey in range(11):
            if eboard[(ex, ey)] == ((train_agent_idx - 10 + 1) % 4) + 10 or eboard[(ex, ey)] == (
                    (train_agent_idx - 10 + 3) % 4) + 10:
                eboard[(ex, ey)] = True
            else:
                eboard[(ex, ey)] = False
    maps.append(eboard)
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*16

# 11*11*12
def featurize_bak(obs):  # no one-hot
    maps = []
    board = obs['board']
    # maps.append(board)

    powerup = board
    ragids = board
    dangers = board

    for ex in range(11):
        for ey in range(11):
            if powerup[ex, ey] not in [6, 7, 8]:
                powerup[ex, ey] = 0
    maps.append(powerup)

    for ex in range(11):
        for ey in range(11):
            if ragids[ex, ey] not in [1, 2]:
                ragids[ex, ey] = 0
    maps.append(ragids)

    for ex in range(11):
        for ey in range(11):
            if dangers[ex, ey] not in [3, 4]:
                dangers[ex, ey] = 0
    maps.append(dangers)

    maps.append(board == 5)

    '''爆炸'''
    bomb_life = obs['bomb_life']
    bomb_blast_strength = obs['bomb_blast_strength']
    flame_life = obs['flame_life']

    # maps.append(bomb_life)
    # maps.append(bomb_blast_strength)
    # maps.append(flame_life)

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
    maps.append(bomb_life)

    '''bomb direction'''
    bomb_moving_direction = obs['bomb_moving_direction']
    maps.append(bomb_moving_direction)

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
            # maps.append(board == e.value

    """训练智能体的位置 one-hot"""
    for idx in [10, 11, 12, 13]:
        if idx not in enemies_idx + [teammate_idx]:
            train_agent_idx = idx
            break
    eboard = board
    for ex in range(11):
        for ey in range(11):
            if eboard[(ex, ey)] == ((train_agent_idx - 10 + 1) % 4) + 10 or eboard[(ex, ey)] == (
                    (train_agent_idx - 10 + 3) % 4) + 10:
                eboard[(ex, ey)] = True
            else:
                eboard[(ex, ey)] = False
    maps.append(eboard)
    maps.append(board == train_agent_idx)

    return np.stack(maps, axis=2)  # 11*11*12

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
    pos = next_pos(obs['position'])
    bomb_life = get_bomb_life(obs)
    for p in pos:
        if bomb_life[p] > 0:
            return True
    return False

def judge_enemy(obs):
    if (obs['board'] == 11).any() or (obs['board'] == 13).any():
        return True
    return False

# def judge_bomb(obs):
#     bomb_life = get_bomb_life(obs)
#     if (bomb_life > 0).any():
#         return True  # 有爆炸威胁
#     else:
#         return False  # 无爆炸威胁

# 判断周围有无爆炸威胁

def next_pos(pos):
    x, y = pos
    up = (np.clip(x - 1, 0, 10), y)
    down = (np.clip(x + 1, 0, 10), y)
    left = (x, np.clip(y - 1, 0, 10))
    right = (x, np.clip(y + 1, 0, 10))
    return [pos, up, down, left, right]
