import numpy as np
import tensorflow as tf
import argparse
import warnings
warnings.filterwarnings("ignore")

from envs import Environment
from agents_value_based import DDQNAgent

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=bool, default=False)
parser.add_argument('--num_episodes', type=int, default=10_000)
parser.add_argument('--name', type=str, default='ddqn-agent')
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

if args.use_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
else:
    tf.config.set_visible_devices([], 'GPU')


# read agent and environment
agent = DDQNAgent(name=args.name)
env = Environment()

# train the agent for num_episodes
agent.train(env=env, num_episodes=args.num_episodes, random_seed=args.seed)

# save trained weights and training data
agent.save_weights()
agent.save_training_data()
