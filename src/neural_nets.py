import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal

RANDOM_SEED = 42

class QNetwork(tf.keras.Model):

    def __init__(self,
                 hidden_sizes=[1024, 1024],
                 dropout_rates=[0.2, 0.2],
                 output_dims=11):

        super(QNetwork, self).__init__()

        self.dense_block = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_sizes[0],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[0], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=hidden_sizes[1],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[1], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=output_dims,
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
        ])

    def call(self, inputs):

        inputs = tf.where(
            inputs >= 0, tf.math.log(tf.math.maximum(inputs, 0.001)), -10)

        return self.dense_block(inputs)


class DuelingNetwork(tf.keras.Model):

    def __init__(self,
                 hidden_sizes=[1024, 1024],
                 dropout_rates=[0.2, 0.2],
                 output_dims=11):

        super(DuelingNetwork, self).__init__()

        self.feat_block = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_sizes[0],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[0], seed=RANDOM_SEED),
        ])

        self.val_block = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_sizes[1],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[1], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=1,
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
        ])

        self.adv_block = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_sizes[1],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[1], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=output_dims,
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
        ])

    def call(self, inputs):
        inputs = tf.where(
            inputs >= 0, tf.math.log(tf.math.maximum(inputs, 0.001)), -10)
        feat = self.feat_block(inputs)
        vals = self.val_block(feat)
        advs = self.adv_block(feat)
        qvals = vals + (advs - tf.math.reduce_mean(advs))
        return qvals


class ActorCriticNetwork(tf.keras.Model):

    def __init__(self,
                 hidden_units=[1024, 1024],
                 dropout_rates=[0.2, 0.2],
                 output_dims=[11, 1]):

        super(ActorCriticNetwork, self).__init__()

        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_units[0],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[0], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=output_dims[0],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('softmax'),
        ])

        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=hidden_units[0],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rates[0], seed=RANDOM_SEED),
            tf.keras.layers.Dense(
                units=output_dims[1],
                kernel_initializer=TruncatedNormal(0.0, 0.05, seed=RANDOM_SEED))
        ])

    def call(self, inputs):
        inputs = tf.where(
            inputs >= 0, tf.math.log(tf.math.maximum(inputs, 0.001)), -10)
        policy_dist = self.actor(inputs)
        value = self.critic(inputs)
        return policy_dist, value
