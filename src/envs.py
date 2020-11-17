import numpy as np
import random
from functools import partial
from abc import ABCMeta, abstractmethod
import itertools

from util import neuekuss_isocratic, nelder_mead


def obtain_compound(instance):

    compute_S1 = lambda x: (
        10**np.random.uniform(0.9, 1.8)
    )
    compute_S2 = lambda x: (
        (np.log10(x) * 2.5009803738203606 - 2.0822057907433917)
        + np.random.uniform(-.35, .35)
    )
    compute_k0 = lambda x: (
        10**((x * 0.08391194231285978 + 0.5054407706077791)
        + np.random.uniform(-1.2, 1.2))
    )

    if instance is not None:
        # if a pandas instance is passed
        retention_factors = instance.to_numpy()
        phi = np.array(instance.index, dtype=np.float32)
        return retention_factors, phi
    else:
        phi = np.array([
            0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90
        ])

        S1 = compute_S1(None)
        S2 = compute_S2(S1)
        k0 = compute_k0(S1)

        noise = 1 + np.random.randn(len(phi)) * 0.1
        retention_factors = neuekuss_isocratic(S1, S2, k0, phi) * noise

        return retention_factors, phi


class Environment:

    def __init__(self, max_num_actions=10):
        self.max_num_actions = min(max_num_actions, 100)

    def _compute_error(self, target, state):
        '''
        Computes the error between predicted retention factors (fitted on the
        scouting runs selected by the RL agent) and experimental/simulated
        retention factors. This error we want to minimize (see objective_fn).
        '''
        def objective_fn(parameters, k, phi):
            return np.mean(np.square((k-neuekuss_isocratic(*parameters, phi))/k))

        obs = np.where(state != -1)[0]

        grid_search = [
                [1, 10     ],     # S1
                [1, 5],           # S2
                [1, 10, 300],     # k0
                [0.1, 1.0, 5.0]   # stepsize
            ]

        grid_search = itertools.product(*grid_search)

        best_score = float('inf')
        for *param, step in grid_search:
            initial_param = np.asarray(param, dtype=np.float32)

            result = nelder_mead(
                partial(objective_fn, k=state[obs], phi=self.phi[obs]),
                initial_param,
                step=step
            )
            if result[1] < best_score:
                best_score = result[1]
                best_param = result[0]

        preds = neuekuss_isocratic(*best_param, self.phi)

        return np.average(
            (np.abs((target - preds)) / target))

    def _compute_reward(self, episode_actions, current_state):
        '''
        Computes the reward based on the selected actions by the RL agent which
        is caculated by dividing 1 with _compute_error (smaller error -> greater
        reward). There are also some additional factors involved in calculating
        the final reward (see below).
        '''
        def sigmoidal(x, m, w, b):
            return 1 / (1 + np.exp( - (x * w + b) )) * m

        error = 1.0
        reward = 0
        terminal = False

        if episode_actions[-1] == 0 or len(episode_actions) == self.max_num_actions:
            terminal = True
            if len(np.where(current_state >= 0)[0]) > 2:
                # first reward: based on mean absolute % error
                error = self._compute_error(self.compound, current_state)
                reward = 1/error
        else:
            k = current_state[episode_actions[-1]-1]
            reward -= sigmoidal(k, 20, 0.00425, -4.0)
            if episode_actions[-1] in episode_actions[:-1]:
                reward -= 5

        return reward, terminal, error

    def reset(self, instance=None, random_initial_step=False):

        self.episode_actions = []
        self.episode_rewards = []

        self.compound, self.phi = obtain_compound(instance)

        # state dim = compound dim = phi dim
        initial_state = -np.ones(len(self.compound), dtype=np.float32)

        if random_initial_step:
            action = np.random.choice(range(1, len(self.compound)+1))
            initial_state[action-1] = self.compound[action-1].copy()

        initial_reward = 0
        terminal = False

        self.state_reward_term = (initial_state, initial_reward, terminal)

        return self.state_reward_term[0]

    def step(self, action):

        previous_state = self.state_reward_term[0]

        current_state = previous_state.copy()

        if action != 0:
            current_state[action-1] = self.compound[action-1].copy()

        self.episode_actions.append(action)

        reward, terminal, error = self._compute_reward(
            self.episode_actions, current_state,)

        if error is not None:
            self.episode_error = np.clip(error, 0, 1)

        self.episode_rewards.append(reward)

        self.state_reward_term = (current_state, reward, terminal)

        return self.state_reward_term

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
