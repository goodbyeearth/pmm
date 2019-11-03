"""The Pommerman v2 Environment, which has communication across the agents.

The communication works by allowing each agent to send a vector of
radio_num_words (default = 2) from a vocabulary of size radio_vocab_size
(default = 8) to its teammate each turn. These vectors are passed into the
observation stream for each agent.
"""
from gym import spaces
import numpy as np

from .. import constants
from .. import utility
from . import v0


class Pomme(v0.Pomme):
    '''The hardest pommerman environment. This class expands env v0
    adding communication between agents.'''
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        self._radio_vocab_size = kwargs.get('radio_vocab_size')
        self._radio_num_words = kwargs.get('radio_num_words')
        if (self._radio_vocab_size and
                not self._radio_num_words) or (not self._radio_vocab_size and
                                               self._radio_num_words):
            assert ("Include both radio_vocab_size and radio_num_words.")

        self._radio_from_agent = {
            agent: (0, 0)
            for agent in [
                constants.Item.Agent0, constants.Item.Agent1,
                constants.Item.Agent2, constants.Item.Agent3
            ]
        }
        super().__init__(*args, **kwargs)

    def _set_action_space(self):
        self.action_space = spaces.Tuple(
            tuple([spaces.Discrete(6)] +
                  [spaces.Discrete(self._radio_vocab_size
                                  )] * self._radio_num_words))

    def _set_observation_space(self):
        """The Observation Space for each agent.

        Total observatiosn: 3*board_size^2 + 12 + radio_vocab_size * radio_num_words:
        - all of the board (board_size^2)
        - bomb blast strength (board_size^2).
        - bomb life (board_size^2)
        - agent's position (2)
        - player ammo counts (1)
        - blast strength (1)
        - can_kick (1)
        - teammate (one of {AgentDummy.value, Agent3.value}).
        - enemies (three of {AgentDummy.value, Agent3.value}).
        - radio (radio_vocab_size * radio_num_words)
        """
        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value
                                            ] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size
                                                ] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        min_obs.extend([0] * self._radio_vocab_size * self._radio_num_words)
        max_obs.extend([1] * self._radio_vocab_size * self._radio_num_words)
        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def get_observations(self):
        self.observations = self.model.get_observations(
            self._board, self._agents, self._bombs, self._flames,
            self._is_partially_observable, self._agent_view_size,
            self._game_type, self._env)
        for obs in self.observations:
            obs['step_count'] = self._step_count
        for obs in self.observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]

        return self.observations

    def get_full_observations(self):
        self.full_observations = self.model.get_full_observations(
            self._board, self._agents, self._bombs, self._flames,
            self._is_partially_observable, self._agent_view_size,
            self._game_type, self._env)
        for obs in self.full_observations:
            obs['step_count'] = self._step_count
        for obs in self.full_observations:
            obs['message'] = self._radio_from_agent[obs['teammate']]
        self.full_observations[0]['my_bomb'] = []
        return self.full_observations


    def reset(self):
        assert (self._agents is not None)

        if self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                pos = np.where(self._board == utility.agent_value(agent_id))
                row = pos[0][0]
                col = pos[1][0]
                agent.set_start_position((row, col))
                agent.reset()
        full_obs = self.get_full_observations()
        obs = self.get_observations()
        obs[0] = full_obs[0]
        return obs

    def act(self, obs):
        agents = [agent for agent in self._agents \
                  if agent.agent_id != self.training_agent]
        return self.model.act(agents, obs, self.action_space)

    def step(self, actions):
        personal_actions = []
        radio_actions = []
        for agent_actions, agent in zip(actions, self._agents):
            if type(agent_actions) == int or not agent.is_alive:
                personal_actions.append(agent_actions)
                radio_actions.append((0, 0))
            elif type(agent_actions) in [tuple, list]:
                personal_actions.append(agent_actions[0])
                radio_actions.append(
                    tuple(agent_actions[1:(1+self._radio_num_words)]))
            else:
                raise

            self._radio_from_agent[getattr(
                constants.Item, 'Agent%d' % agent.agent_id)] = radio_actions[-1]

        self._intended_actions = personal_actions

        # get previous state
        old_state = self.get_observations()


        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(
            personal_actions,
            self._board,
            self._agents,
            self._bombs,
            self._items,
            self._flames,
            max_blast_strength=max_blast_strength)
        self._board, self._agents, self._bombs, self._items, self._flames = \
            result[:5]

        done = self._get_done()
        obs = self.get_observations()
        curr_full_obs = self.get_full_observations()
        # reward = self._get_rewards()
        reward1 = self._get_rewards1()
        reward2 = self._get_rewards2()
        # reward3 = self._get_rewards3(obs,old_state)
        # reward4 = self._get_rewards4(obs,old_state)
        reward = reward1
        # reward = list(np.array(reward1)+np.array(reward2)+np.array(reward3))
        info = self._get_info(done, reward)

        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

        self._step_count += 1

        # change to full obs
        # obs[0] = curr_full_obs[0]
        return obs, reward, done, info
        # return super().step(personal_actions)

    # self define reward function
    # need to modify the return reward
    # now is for agent0
    def _get_rewards1(self):
        return self.model.get_rewards1(self._agents, self._game_type, self._step_count, self._max_steps,0)
    def _get_rewards2(self):
        return self.model.get_rewards2(self._agents, self._game_type, self._step_count, self._max_steps,0)
    # def _get_rewards3(self,curr_state,old_state):
    #     return self.model.get_rewards3(self._agents, self._game_type, self._step_count, self._max_steps,curr_state,old_state,0)
    # def _get_rewards4(self,curr_state,old_state,):
    #     return self.model.get_rewards4(self._agents, self._game_type, self._step_count,
    #                                    self._max_steps,curr_state,old_state,0)

    @staticmethod
    def featurize(obs):
        ret = super().featurize(obs)
        message = obs['message']
        message = utility.make_np_float(message)
        return np.concatenate((ret, message))

    def get_json_info(self):
        ret = super().get_json_info()
        ret['radio_vocab_size'] = json.dumps(
            self._radio_vocab_size, cls=json_encoder)
        ret['radio_num_words'] = json.dumps(
            self._radio_num_words, cls=json_encoder)
        ret['_radio_from_agent'] = json.dumps(
            self._radio_from_agent, cls=json_encoder)
        return ret

    def set_json_info(self):
        super().set_json_info()
        self.radio_vocab_size = json.loads(
            self._init_game_state['radio_vocab_size'])
        self.radio_num_words = json.loads(
            self._init_game_state['radio_num_words'])
        self._radio_from_agent = json.loads(
            self._init_game_state['_radio_from_agent'])
