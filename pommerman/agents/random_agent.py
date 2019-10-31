'''An agent that preforms a random action each step'''
from . import BaseAgent

class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)
        self.flag = 0

    def act(self, obs, action_space):
        if self.flag == 0:
            self.flag += 1
            return 5
        elif self.flag == 1:
            self.flag += 1
            return 4
        else:
            return 5
        # return (1,1,1)
