'''An agent that preforms a random action each step'''
from . import BaseAgent

class StopAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(StopAgent, self).__init__(*args, **kwargs)


    def act(self, obs, action_space):
        return 0
