'''An agent that preforms a random action each step'''
from . import BaseAgent

class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.flag = 0

    def act(self, obs, action_space):
        # return action_space.sample()
        if(self.flag == 0):
            act = 5
        if (self.flag == 1):
            act = 3
        if (self.flag == 2):
            act = 5
        if (self.flag == 3):
            act = 3
        if (self.flag == 4):
            act = 2
        if (self.flag >= 5):
            act = 5
        self.flag += 1
        return act
