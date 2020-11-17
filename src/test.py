import numpy as np
import tensorflow as tf
import warnings
import itertools
import functools

from agents_value_based import DDQNAgent
from util import neuekuss_isocratic, nelder_mead

tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

def fit(model, x, phi):

    def objective_fn(parameters, k, phi):
        return np.mean(np.square((k-model(*parameters, phi))/k))

    grid_search = [
            [1, 10, 300],     # k0
            [1, 10     ],     # S1
            [1, 5],           # S2
            [0.1, 1.0, 5.0]   # stepsize
        ]

    grid_search = itertools.product(*grid_search)

    best_score = float('inf')
    for *param, step in grid_search:
        initial_param = np.asarray(param, dtype=np.float32)

        result = nelder_mead(
            functools.partial(objective_fn, k=x, phi=phi),
            initial_param,
            step=step
        )
        if result[1] < best_score:
            best_score = result[1]
            best_param = result[0]

    return best_param

agent = DDQNAgent(name='ddqn-agent', tf_summary_writer=False)
try:
    agent.load_weights()
except:
    print("\nTrained agent not found ... untrained agent used instead")

mappings = {
    0:  (0.00, 'STOP'),
    1:  (0.05, '5%' ),
    2:  (0.10, '10%'),
    3:  (0.20, '20%'),
    4:  (0.30, '30%'),
    5:  (0.40, '40%'),
    6:  (0.50, '50%'),
    7:  (0.60, '60%'),
    8:  (0.70, '70%'),
    9:  (0.80, '80%'),
    10: (0.90, '90%'),
}


input('\nHit enter to start')
s = -np.ones(10, dtype=np.float32)
a = np.argmax(agent.model(s[np.newaxis])[0])
while True:

    if a == 0:
        print('Agent selects to stop performing scouting runs ' +
              '...\n... scouting runs finished!')

        phi = np.array([mappings[i][0] for i in np.where(s != -1)[0]+1])
        S1, S2, kw = fit(neuekuss_isocratic,  s[np.where(s != -1)[0]], phi)

        print('Model estimated to be:\n' +
              'k(phi) = {:.3f} * (1 + {:.3f} * phi)^2 * np.exp(-({:.3f} * phi)/(1 + {:.3f} * phi))'
              .format(kw, S2, S1, S2))

        break

    print('Agent selects scouting run at {} ACN'.format(mappings[a][1]))
    k = input('Feed retention factor value of scouting run at {} ACN: '.format(mappings[a][1]))
    s[a-1] = k

    a = np.argmax(agent.model(s[np.newaxis])[0])
