import numpy as np
import tensorflow as tf
import glob
import shutil
import collections
from pathlib import Path

from definitions import OUTPUT_DIR, WEIGHTS_DIR, LOGS_DIR


class AgentMixin:

    '''
    This class is mixed in with the Agent Classes, and is mainly to make it more
    convenient to save/load models and to run the trained RL agent on experimental
    data after it has been trained.

    Note that "self.name" comes from the Agent Classes that it's mixed with.
    '''

    @classmethod
    def from_pretrained(cls, filename):
        return cls(pretrained_weights=WEIGHTS_DIR+filename)

    def save_training_data(self):
        np.save('{}{}-errors.npy'.format(
            OUTPUT_DIR, self.name), self.errors)
        np.save('{}{}-rewards.npy'.format(
            OUTPUT_DIR, self.name), self.rewards)
        np.save('{}{}-num_actions.npy'.format(
            OUTPUT_DIR, self.name), self.num_actions)

    def save_weights(self, path=None):
        if path is None:
            wdir = Path(WEIGHTS_DIR)
            wdir.mkdir(exist_ok=True)
            self.model.save_weights("{}/{}.h5".format(wdir, self.name))
        else:
            self.model.save_weights(path)

    def load_weights(self, path=None):
        if path is None:
            self.model.load_weights(WEIGHTS_DIR+"{}.h5".format(self.name))
        else:
            self.model.load_weights(path)
        if hasattr(self, 'target_model'):
            self.target_model.set_weights(self.model.get_weights())

    def create_tf_summary_writer(self):
        files = glob.glob(LOGS_DIR+'{}*'.format(self.name))
        for f in files:
            shutil.rmtree(f)
        self.tf_summary_writer = tf.summary.create_file_writer(
            LOGS_DIR+'{}'.format(self.name)
        )

    def perform_task(self, env, compound, random_initial_step=False):
        record = collections.defaultdict(list)
        current_state = env.reset(compound, random_initial_step)
        while 1:
            value = self.model(current_state[np.newaxis], training=False)
            if len(value) != 1:
                value = value[0][0].numpy()
            else:
                value = value[0].numpy()
            action = np.argmax(value)
            next_state, reward, done = env.step(action)
            record['v'].append(value)
            record['a'].append(action)
            record['s'].append(current_state)
            record['r'].append(reward)
            current_state = next_state
            if done:
                break
        return record
