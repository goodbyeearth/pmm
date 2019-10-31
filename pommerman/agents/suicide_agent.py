'''An agent that preforms a random action each step'''
from . import BaseAgent

class SuicideAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        # return action_space.sample()
        return 5
        # return (1,1,1)