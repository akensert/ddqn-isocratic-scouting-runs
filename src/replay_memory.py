import numpy as np
from collections import deque
from abc import ABCMeta, abstractmethod


class BaseReplayMemory(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, capacity = -1):
        self.memory = deque()
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)

    def _obtain_transitions(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for index in indices:
            instance = self.memory[index]
            states.append(instance[0])
            actions.append(instance[1])
            rewards.append(instance[2])
            next_states.append(instance[3])
            dones.append(instance[4])

        return (states, actions, rewards, next_states, dones)

    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass


class ReplayMemory(BaseReplayMemory):

    def __init__(self, capacity = -1):
        super().__init__(capacity=capacity)

    def add(self, transition):
        self.memory.append(transition)
        if self.capacity != -1 and len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        transitions = self._obtain_transitions(indices)
        return transitions


class PrioritizedReplayMemory(BaseReplayMemory):

    def __init__(self, capacity = -1,
                 e = 1e-3,
                 alpha = 1.0,
                 beta = 0.0,
                 beta_incr = 0.0001):

        super().__init__(capacity=capacity)

        self.priority = deque()
        self.e = e
        self.alpha = alpha
        self.beta = beta
        self.beta_incr = beta_incr

    def add(self, transition, abs_td_error):
        priority = (abs_td_error + self.e) ** self.alpha
        self.memory.append(transition)
        self.priority.append(priority)
        if self.capacity != -1 and len(self.memory) > self.capacity:
            self.memory.popleft()
            self.priority.popleft()

    def update(self, indices, abs_td_error):
        priority = (abs_td_error + self.e) ** self.alpha
        for i, index in enumerate(indices):
            self.priority[index] = priority[i]

    def sample(self, batch_size):

        # compute propability of sampling an instance
        #    based on priority values (= TD errors)
        psum = np.sum(self.priority)
        prob = self.priority / psum

        indices = np.random.choice(len(prob), size=batch_size, p=prob)

        # compute importance, which will weight the loss function (sample-wise)
        importance = ((1/prob) * (1/len(self.priority)))**self.beta
        # identical to np.power(len(self.priority) * prob, -self.beta)
        importance = np.array(importance)[indices]
        # importance /= importance.max()

        self.beta = min(1., self.beta + self.beta_incr)

        transitions = self._obtain_transitions(indices)

        return transitions, importance, indices
