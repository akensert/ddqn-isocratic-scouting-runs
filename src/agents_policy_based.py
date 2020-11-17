

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import tqdm

from neural_nets import ActorCriticNetwork
from losses import actor_critic_loss
from mixin import AgentMixin


class BaseA2CAgent(AgentMixin, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, name, model, loss_fn,
                 optimizer, learning_rate,
                 gamma, critic_discount,
                 entropy_beta, input_shape,
                 output_dims, pretrained_weights,
                 tf_summary_writer=True,):

        self.name = name

        if tf_summary_writer:
            self.create_tf_summary_writer()

        self.model = model(output_dims=output_dims)
        self.model.build(input_shape=input_shape)

        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

        # optimizer and loss function
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
           initial_learning_rate=learning_rate,
           decay_steps=1024,
           end_learning_rate=learning_rate * 0.1,
           power=1,
           cycle=False,
        )
        self.loss_fn = loss_fn
        self.optimizer = optimizer(learning_rate)

        self.gamma = gamma
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta


    def _train(self, env, num_episodes, num_steps):

        # initialize accumulators
        self.errors = []
        self.num_actions = []
        self.rewards = []

        best_reward = float('-inf')

        state = env.reset()

        pbar = tqdm.tqdm(range(num_episodes), desc=' episodes')
        while 1:

            actions, rewards, values, dones = np.empty((4, num_steps), dtype=np.float32)
            states = np.empty((num_steps, len(env.compound)), dtype=np.float32)

            for step in range(num_steps):
                policy_dist, value = self.model(state[np.newaxis], training=True)
                policy_dist, value = policy_dist.numpy()[0], value.numpy()[0]

                action = np.random.choice(len(policy_dist), p=policy_dist)

                actions[step] = action
                values[step] = value

                next_state, reward, done = env.step(action)

                rewards[step] = reward
                dones[step] = int(done)
                states[step] = state

                if done:
                    if self.tf_summary_writer:
                        with self.tf_summary_writer.as_default():
                            tf.summary.scalar('errors', env.episode_error, step=pbar.n)
                            tf.summary.scalar('rewards', np.sum(env.episode_rewards), step=pbar.n)
                            tf.summary.scalar('num_actions', len(env.episode_actions), step=pbar.n)

                    self.errors.append(env.episode_error)
                    self.num_actions.append(len(env.episode_actions))
                    self.rewards.append(np.sum(env.episode_rewards))
                    state = env.reset()
                    pbar.update(1)
                    pbar.set_description(
                      "actions {:.3f} : rewards {:.3f}".format(
                        np.mean(self.num_actions[-100:])-1, np.mean(self.rewards[-100:])))
                else:
                    state = next_state.copy()

            _, value = self.model(state[np.newaxis], training=True)

            q_values = np.append(np.zeros_like(rewards), value.numpy()[0], axis=-1)
            # Qt = rt+1 + gamma * Vt+1
            for t in reversed(range(len(rewards))):
                q_values[t] = rewards[t] + self.gamma * q_values[t+1] * (1 - dones[t])
            q_values = q_values[:-1]

            advantages = q_values - values

            with tf.GradientTape() as tape:
                policy_dists, values = self.model(states, training=True)
                loss = self.loss_fn(
                    y_true=[(actions, advantages), q_values], y_pred=[policy_dists, values],
                    critic_discount=self.critic_discount, entropy_beta=self.entropy_beta)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            if pbar.n >= num_episodes:
                break


class A2CAgent(BaseA2CAgent):

    def __init__(self,
                 name='A2C',
                 model=ActorCriticNetwork,
                 loss_fn=actor_critic_loss,
                 optimizer=tf.keras.optimizers.Adam,
                 learning_rate=1e-4,
                 gamma=0.9,
                 critic_discount=0.5,
                 entropy_beta=0.0001,
                 input_shape=(None, 10),
                 output_dims=[11, 1],
                 pretrained_weights=None):

        super().__init__(
            name=name,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            gamma=gamma,
            critic_discount=critic_discount,
            entropy_beta=entropy_beta,
            input_shape=input_shape,
            output_dims=output_dims,
            pretrained_weights=pretrained_weights,)

    def train(self, env, num_episodes, num_steps):
        self._train(env, num_episodes, num_steps)
