import numpy as np
import pandas as pd
import copy
from pathlib import Path
import tensorflow as tf
import itertools
import functools
import multiprocessing
import shutil
import glob
import warnings
import math

from definitions import DATA_DIR


def neuekuss_isocratic(S1, S2, k0, phi):
    '''
    Neue Kuss formula for isocratic models
    '''
    return k0 * np.square(1 + S2 * phi) * np.exp(-(S1 * phi)/(1 + S2 * phi))

def neuekuss_gradient(S1, S2, k0, phi_start, phi_end, tG, tD, t0):
    '''
    Neue Kuss formula for gradient models
    '''
    beta = (phi_end - phi_start) / tG
    k_start = neuekuss_isocratic(S1, S2, k0, phi_start)
    phi_elution = (
        ((1 + S2 * phi_start)**2/S1 *
         np.log(1 + beta * k0 * S1 * (t0 - tD/k_start) *
                np.exp((-S1 * phi_start)/(1 + S2*phi_start)))) /
        (1 - (S2 * (1 + S2 * phi_start))/S1 *
         np.log(1 + beta * k0 * S1 * (t0 - tD/k_start) *
                np.exp((-S1 * phi_start)/(1 + S2*phi_start))))
    )
    return tD/t0 + phi_elution / (beta*t0)

def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)

    Source code and License found at https://github.com/fchollet/nelder-mead

    All credit goes to https://github.com/fchollet/
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        # print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


def read_data(filename, threshold=0.001, fillnan=True):

    def impute_values(model, state, phi, nanwhere):
        '''
        Used only with read_data()
        '''
        def loss(parameters, k, phi):
            return np.mean(np.square(k-model(*parameters, phi))/k)

        assert len(state) == len(phi)

        grid_search = [
                [1, 10     ],
                [1, 5],
                [1, 10, 100],
                [0.1, 1.0, 5.0]
            ]

        grid_search = itertools.product(*grid_search)

        best_score = float('inf')
        for *param, step in grid_search:
            initial_param = np.asarray(param, dtype=np.float32)

            result = nelder_mead(
                functools.partial(loss, k=state[~nanwhere], phi=phi[~nanwhere]),
                initial_param,
                step=step
            )
            if result[1] < best_score:
                best_score = result[1]
                best_param = result[0]

        return model(*best_param, phi[nanwhere])

    def fill_nan(data):
        '''
        Used only with read_data()
        '''
        phi = np.array([
            0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90
        ])
        for i, row in data.iloc[:, :10].iterrows():
            nanwhere = np.isnan(row)
            if nanwhere.any() > 0:
                data.loc[i].iloc[:10][nanwhere] = impute_values(
                    neuekuss_isocratic, row.values, phi, nanwhere)
        return data


    data = pd.read_excel(DATA_DIR+filename, index_col=0)
    data[data<threshold] = threshold
    if fillnan:
        data = fill_nan(data)
    #else:
        #data.dropna(inplace=True)
    return data

def configure(disable_gpu=False, filter_warnings=False):
    if filter_warnings:
        warnings.filterwarnings("ignore")

    np.set_printoptions(suppress=True)

    print("> TensorFlow version", tf.__version__)

    if disable_gpu:
        tf.config.set_visible_devices([], 'GPU')
        cpus = gpus = tf.config.experimental.list_physical_devices('CPU')
        if cpus:
            logical_cpus = tf.config.threading.get_intra_op_parallelism_threads()
            print("  > Running TensorFlow on available CPU(s)")
            print("  > Found", len(cpus), "Physical CPU(s),", multiprocessing.cpu_count(), "CPU threads")
    else:
        print("  > Running TensorFlow on available GPU(s)")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("  > Found", len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
            except RuntimeError as e:
                print(e)

def cosine_epsilon(
    episode,
    total_steps,
    cycles=10,
    reduction=0.0,
    eps_min=0.1,
    eps_max=1.0):

    cycles = cycles - 0.5
    cycle_steps = (total_steps / cycles)

    step = episode / cycle_steps * 2

    if (episode-cycle_steps//2) % int(cycle_steps) == 0:
        cycle_minimum = True
    else:
        cycle_minimum = False

    reduction = max(1 - (episode * (reduction / cycle_steps)), 0)

    eps = (eps_max * 0.5 - eps_min/2) * reduction * (1.0 + math.cos(math.pi * step)) + eps_min

    return eps
