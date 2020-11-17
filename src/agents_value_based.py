import numpy as np

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import tqdm

from neural_nets import DuelingNetwork, QNetwork
from replay_memory import ReplayMemory, PrioritizedReplayMemory
from losses import weighted_mean_squared_error
from mixin import AgentMixin


class BaseDDQNAgent(AgentMixin, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, name, model, replay_memory,
                 loss_fn, optimizer, batch_size,
                 learning_rate, update_target_network_every,
                 update_prediction_network_every, replay_memory_capacity,
                 batches_before_training, gamma, eps_initial, eps_minimum,
                 eps_decay, input_shape, output_dims, pretrained_weights,
                 tf_summary_writer):

        self.name = name

        if tf_summary_writer:
            self.create_tf_summary_writer()

        # prediction network
        self.model = model(output_dims=output_dims)
        self.model.build(input_shape=input_shape)
        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

        #print(self.model.trainable_weights[0][0][0])
        # target network
        self.target_model = model(output_dims=output_dims)
        self.target_model.build(input_shape=input_shape)
        self.target_model.set_weights(self.model.get_weights())

        # optimizer and loss function
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
           initial_learning_rate=learning_rate,
           decay_steps=4096,
           end_learning_rate=learning_rate * 0.1,
           power=1,
           cycle=False,
        )
        self.optimizer = optimizer(learning_rate_fn)
        self.loss_fn = loss_fn

         # replay memory
        self.replay_memory = replay_memory(capacity=replay_memory_capacity)

        self.batches_before_training = batches_before_training

        self.batch_size = batch_size
        self.update_target_network_every = update_target_network_every
        self.update_prediction_network_every = update_prediction_network_every

        self.gamma = gamma
        self.epsilon = eps_initial
        self.eps_minimum = eps_minimum
        self.eps_decay = eps_decay


    def _obtain_batch(self):
        batch = self.replay_memory.sample(self.batch_size)

        if len(batch) == 3:
            # if prioritized replay memory
            batch, importance, indices = batch
        else:
            # if vanilla replay memory
            importance, indices = None, None

        states = tf.convert_to_tensor(batch[0], dtype=tf.float32)
        actions = tf.convert_to_tensor(batch[1], dtype=tf.int32)
        rewards = tf.convert_to_tensor(batch[2], dtype=tf.float32)
        next_states = tf.convert_to_tensor(batch[3], dtype=tf.float32)
        dones = tf.convert_to_tensor(batch[4], dtype=tf.float32)

        return (
            (states, actions, rewards, next_states, dones), importance, indices
        )

    def _update_prediction_network(self):

        (states, actions, rewards, next_states, dones), importance, indices = \
            self._obtain_batch()

        with tf.GradientTape() as tape:
            # current states -> current Q-values
            current_Q = self.model(states, training=True)

            # reduce Q to an 1-d array corresponding to the actions taken
            idx_to_gather = tf.stack(
                [tf.range(actions.shape[0]), actions], axis=1)
            current_Q = tf.gather_nd(current_Q, idx_to_gather)

            # next states -> next Q-values
            expected_Q = self.target_model(next_states).numpy()

            # Q learning: obtain target Q-values
            expected_Q = (
                rewards + self.gamma
                * tf.math.reduce_max(expected_Q, axis=1)
                * (1 - dones)
            )

            loss = self.loss_fn(current_Q, expected_Q, importance)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        if hasattr(self.replay_memory, 'update'):
            abs_td_error = np.abs((expected_Q - current_Q).numpy())
            self.replay_memory.update(indices, abs_td_error)

    def _train(self, env, num_episodes, random_seed):

        tf.random.set_seed(random_seed)

        best_reward = float('-inf')

        # initialize accumulators
        self.errors = []
        self.num_actions = []
        self.rewards = []
        self.td_errors = []

        pbar = tqdm.tqdm(range(num_episodes), desc=' episodes')
        for _ in pbar:

            np.random.seed(random_seed)
            random_seed += 1

            state = env.reset()

            episode_td_error = []
            while True:

                q_values = self.model(state[np.newaxis], training=True)

                q_values = np.squeeze(q_values.numpy())
                if np.random.random() > self.epsilon:
                    action = np.argmax(q_values)
                else:
                    action = np.random.choice(len(q_values))

                next_state, reward, done = env.step(action)

                current_Q = q_values[action]
                expected_Q = self.target_model(next_state[np.newaxis]).numpy()
                expected_Q = reward + self.gamma * np.max(expected_Q) * (1 - int(done))
                TD_error = np.abs(expected_Q - current_Q)

                if hasattr(self.replay_memory, 'priority'):
                    self.replay_memory.add((state, action, reward, next_state, done), TD_error)
                else:
                    self.replay_memory.add((state, action, reward, next_state, done))

                episode_td_error.append(TD_error)

                if done:
                    if self.tf_summary_writer:
                        with self.tf_summary_writer.as_default():
                            tf.summary.scalar('td_error', np.mean(episode_td_error), step=pbar.n)
                            tf.summary.scalar('errors', env.episode_error, step=pbar.n)
                            tf.summary.scalar('rewards', np.sum(env.episode_rewards), step=pbar.n)
                            tf.summary.scalar('num_actions', len(env.episode_actions), step=pbar.n)
                            tf.summary.scalar('epsilon', self.epsilon, step=pbar.n)
                    self.errors.append(env.episode_error)
                    self.num_actions.append(len(env.episode_actions))
                    self.rewards.append(np.sum(env.episode_rewards))
                    self.td_errors.append(np.mean(episode_td_error))
                    pbar.set_description(
                      "actions {:.3f} : rewards {:.3f}".format(
                        np.mean(self.num_actions[-100:])-1, np.mean(self.rewards[-100:])))
                    break
                else:
                    state = next_state.copy()

            if len(self.replay_memory) >= self.batch_size * self.batches_before_training:

                if pbar.n % self.update_prediction_network_every == 0:
                    self._update_prediction_network()

                if pbar.n % self.update_target_network_every == 0:
                    self.target_model.set_weights(self.model.get_weights())

            self.epsilon = max(self.epsilon * self.eps_decay, self.eps_minimum)


class DDQNAgent(BaseDDQNAgent):

    def __init__(self,
                 name='DDQN',
                 model=QNetwork,
                 replay_memory=ReplayMemory,
                 loss_fn=weighted_mean_squared_error,
                 optimizer=tf.keras.optimizers.SGD,
                 batch_size=128,
                 learning_rate=1e-3,
                 update_target_network_every=64,
                 update_prediction_network_every=1,
                 replay_memory_capacity=2048,
                 batches_before_training=4,
                 gamma=0.95,
                 eps_initial=1.0,
                 eps_minimum=0.1,
                 eps_decay=0.99,
                 input_shape=(None, 10),
                 output_dims=11,
                 pretrained_weights=None,
                 tf_summary_writer=True,):

        super().__init__(
            name=name, model=model, replay_memory=replay_memory,
            loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, learning_rate=learning_rate,
            update_target_network_every=update_target_network_every,
            update_prediction_network_every=update_prediction_network_every,
            replay_memory_capacity=replay_memory_capacity,
            batches_before_training=batches_before_training,
            gamma=gamma, eps_initial=eps_initial, eps_minimum=eps_minimum,
            eps_decay=eps_decay, input_shape=input_shape,
            output_dims=output_dims, pretrained_weights=pretrained_weights,
            tf_summary_writer=tf_summary_writer)

    def train(self, env, num_episodes, random_seed=600):
        self._train(env=env, num_episodes=num_episodes, random_seed=random_seed)


class PERDDQNAgent(BaseDDQNAgent):

    def __init__(self,
                 name='PDDQN',
                 model=QNetwork,
                 replay_memory=PrioritizedReplayMemory,
                 loss_fn=weighted_mean_squared_error,
                 optimizer=tf.keras.optimizers.SGD,
                 batch_size=128,
                 learning_rate=1e-3,
                 update_target_network_every=64,
                 update_prediction_network_every=1,
                 replay_memory_capacity=2048,
                 batches_before_training=4,
                 gamma=0.95,
                 eps_initial=1.0,
                 eps_minimum=0.1,
                 eps_decay=0.99,
                 input_shape=(None, 10),
                 output_dims=11,
                 pretrained_weights=None,
                 tf_summary_writer=True,):

        super().__init__(
            name=name, model=model, replay_memory=replay_memory,
            loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, learning_rate=learning_rate,
            update_target_network_every=update_target_network_every,
            update_prediction_network_every=update_prediction_network_every,
            replay_memory_capacity=replay_memory_capacity,
            batches_before_training=batches_before_training,
            gamma=gamma, eps_initial=eps_initial, eps_minimum=eps_minimum,
            eps_decay=eps_decay, input_shape=input_shape,
            output_dims=output_dims, pretrained_weights=pretrained_weights,
            tf_summary_writer=tf_summary_writer)

    def train(self, env, num_episodes, random_seed=600):
        self._train(env=env, num_episodes=num_episodes, random_seed=random_seed)


class DuelingDDQNAgent(BaseDDQNAgent):

    def __init__(self,
                 name='DDDQN',
                 model=DuelingNetwork,
                 replay_memory=ReplayMemory,
                 loss_fn=weighted_mean_squared_error,
                 optimizer=tf.keras.optimizers.SGD,
                 batch_size=128,
                 learning_rate=1e-3,
                 update_target_network_every=64,
                 update_prediction_network_every=1,
                 replay_memory_capacity=2048,
                 batches_before_training=4,
                 gamma=0.95,
                 eps_initial=1.0,
                 eps_minimum=0.1,
                 eps_decay=0.99,
                 input_shape=(None, 10),
                 output_dims=11,
                 pretrained_weights=None,
                 tf_summary_writer=True,):

        super().__init__(
            name=name, model=model, replay_memory=replay_memory,
            loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, learning_rate=learning_rate,
            update_target_network_every=update_target_network_every,
            update_prediction_network_every=update_prediction_network_every,
            replay_memory_capacity=replay_memory_capacity,
            batches_before_training=batches_before_training,
            gamma=gamma, eps_initial=eps_initial, eps_minimum=eps_minimum,
            eps_decay=eps_decay, input_shape=input_shape,
            output_dims=output_dims, pretrained_weights=pretrained_weights,
            tf_summary_writer=tf_summary_writer)

    def train(self, env, num_episodes, random_seed=600):
        self._train(env=env, num_episodes=num_episodes, random_seed=random_seed)


class PERDuelingDDQNAgent(BaseDDQNAgent):

    def __init__(self,
                 name='PDDDQN',
                 model=DuelingNetwork,
                 replay_memory=PrioritizedReplayMemory,
                 loss_fn=weighted_mean_squared_error,
                 optimizer=tf.keras.optimizers.SGD,
                 batch_size=128,
                 learning_rate=1e-3,
                 update_target_network_every=64,
                 update_prediction_network_every=1,
                 replay_memory_capacity=2048,
                 batches_before_training=4,
                 gamma=0.95,
                 eps_initial=1.0,
                 eps_minimum=0.1,
                 eps_decay=0.99,
                 input_shape=(None, 10),
                 output_dims=11,
                 pretrained_weights=None,
                 tf_summary_writer=True,):

        super().__init__(
            name=name, model=model, replay_memory=replay_memory,
            loss_fn=loss_fn, optimizer=optimizer,
            batch_size=batch_size, learning_rate=learning_rate,
            update_target_network_every=update_target_network_every,
            update_prediction_network_every=update_prediction_network_every,
            replay_memory_capacity=replay_memory_capacity,
            batches_before_training=batches_before_training,
            gamma=gamma, eps_initial=eps_initial, eps_minimum=eps_minimum,
            eps_decay=eps_decay, input_shape=input_shape,
            output_dims=output_dims, pretrained_weights=pretrained_weights,
            tf_summary_writer=tf_summary_writer)

    def train(self, env, num_episodes, random_seed=600):
        self._train(env=env, num_episodes=num_episodes, random_seed=random_seed)
