import numpy as np
import tensorflow as tf


def weighted_mean_squared_error(y_true, y_pred, sample_weights):

    error = tf.math.squared_difference(y_true, y_pred)

    if sample_weights is not None:
        return tf.math.reduce_mean(error * sample_weights)

    return tf.math.reduce_mean(error)


def actor_critic_loss(y_true, y_pred, critic_discount, entropy_beta):

    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    cce = tf.keras.losses.CategoricalCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    # policy loss (a.k.a. actor loss)
    actions, advantages = y_true[0][0], y_true[0][1]#tf.split(y_true[0], 2, axis=-1)
    policy_loss = scce(actions, y_pred[0], sample_weight=advantages)

    # value loss (a.k.a. critic loss)
    value_loss = critic_discount * mse(y_true[1], tf.squeeze(y_pred[1]))

    # entropy loss
    entropy_loss = entropy_beta * cce(y_pred[0], y_pred[0])

    # final loss
    return policy_loss - entropy_loss + value_loss
