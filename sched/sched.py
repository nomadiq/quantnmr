import numpy as np
import matplotlib.pyplot as plt
import os
from subprocess import Popen, PIPE
import random
from collections import OrderedDict


def hyberts_schedule(X, Y, p):
    # determine number of samples from portion
    XY_p = int(X * Y * p)

    # make a schedule from poissonv3 program
    schedule_return = Popen(
        f"poissonv3 2 0 2 {XY_p} 0.0001 {X} {Y} 0 0", shell=True, stdout=PIPE
    )
    schedule = schedule_return.stdout.read().decode()
    schedule = schedule.split("\n")
    dummy = schedule.pop(-1)

    # Populate a matrix with zeros and fill matrix with 1s at sample sites
    sample_matrix = np.zeros((X, Y))
    for sample in schedule:
        x, y = sample.split()
        sample_matrix[int(x)][int(y)] = 1

    # return the full schedule matrix
    return sample_matrix


def sample_matrix_to_schedule(sample_matrix):
    # we use these to store a list with sample lines in them for PGS and the rest of the samples
    X = sample_matrix.shape[0]
    Y = sample_matrix.shape[1]
    pgs_schedule = []
    rest_schedule = []

    # go through entire sample space creating PGS lines and rest lines
    for y in range(Y):
        for x in range(X):
            if sample_matrix[x][y] == 0:
                rest_schedule.append(f"{x}\t{y}")
            else:
                pgs_schedule.append(f"{x}\t{y}")

    # shuffle the rest schedule lines randomly
    random.shuffle(rest_schedule)

    # keep 0,0 line
    temp = pgs_schedule.pop(0)
    # shuffle rest of PGS samples
    random.shuffle(pgs_schedule)
    # add 0,0 to the shuffle
    pgs_schedule = [temp] + pgs_schedule

    # return it
    return pgs_schedule + rest_schedule


# gaps
def sine_g(x, t, N):
    return np.sin(np.pi * t * x / N)


def flat_g(x, t, N):
    return N / t


# weights
def flat_w(x, t, N):
    if isinstance(x, np.ndarray):
        return np.ones(N + 1)
    else:
        return 1


def exp_w(x, t, N):
    return np.exp(-t * x / N)


def cos_w(x, t, N):
    return np.cos(np.pi * t * x / N)


# for weighted schedules this returns a tensor of weights
# with same dimension as number of input weights (using outer product)
def weight_constructor(weights):
    if len(weights) == 1:
        return np.asarray(weights[0])
    outer = weights[0]
    for i in range(1, len(weights)):
        outer = np.multiply.outer(outer, weights[i])
    return outer


def gap_sched(weight_func, params):
    np.random.seed()
    # points = params[0] # this is not used for gap weights
    N = params[0]
    num_selec = params[1]
    param = params[2]
    w = 2.0
    i = 0
    s = []

    while len(s) != num_selec + 1:
        s.append(0)  # always select
        i += 1
        # loop through making samples
        while i < N + 1:
            i += np.random.poisson(
                (N / num_selec) * w * (weight_func(i, param, N - 1))
            )
            s.append(i)
            i += 1

        # test samples - too many?
        if len(s) > num_selec + 1:
            w *= 1.02
            s = []  # reset sample list
            i = 0  # reset sample pointer

        elif len(s) < num_selec + 1:
            w /= 1.02
            s = []  # reset sample list
            i = 0  # reset sample pointer

        elif s[-1] < N:
            s = []  # not weights fault - reset sample list
            i = 0  # reset sample pointer

    s = np.asarray(s[:-1])
    return s, w


def robson_schedule(M, N, sampling_density, gap_func, param, normal=True):
    space = M * N
    subspace = int(space * sampling_density)
    indices = gap_sched(gap_func, [space, subspace, param])[0]

    pythagoras = OrderedDict()
    manhattan = OrderedDict()

    for i in range(M):
        for j in range(N):
            if normal:
                pythagoras[str(i) + "," + str(j)] = np.sqrt(
                    (i / (M - 1)) ** 2 + (j / (N - 1)) ** 2
                )
                manhattan[str(i) + "," + str(j)] = i / (M - 1) + j / (N - 1)
            else:
                pythagoras[str(i) + "," + str(j)] = np.sqrt((i) ** 2 + (j) ** 2)
                manhattan[str(i) + "," + str(j)] = i + j

    ordered_pythagoras = {
        k: v for k, v in sorted(pythagoras.items(), key=lambda item: item[1])
    }
    ordered_manhattan = {
        k: v for k, v in sorted(manhattan.items(), key=lambda item: item[1])
    }
    pythagoras_keys = list(ordered_pythagoras.keys())
    manhattan_keys = list(ordered_manhattan.keys())

    samples_p_x = []
    samples_p_y = []
    samples_m_x = []
    samples_m_y = []

    for i in indices:
        coord = pythagoras_keys[int(i)].split(",")
        samples_p_x.append(int(coord[0]))
        samples_p_y.append(int(coord[1]))

        coord = manhattan_keys[int(i)].split(",")
        samples_m_x.append(int(coord[0]))
        samples_m_y.append(int(coord[1]))

    ps = list(zip(samples_p_x, samples_p_y))
    ms = list(zip(samples_m_x, samples_m_y))

    ps_matrix = np.zeros((M, N))
    ms_matrix = np.zeros((M, N))

    for sample in ps:
        ps_matrix[sample[0]][sample[1]] = 1

    for sample in ms:
        ms_matrix[sample[0]][sample[1]] = 1

    return ps_matrix, ms_matrix


def mobli_schedule(X, Y, t1, t2, p):
    x_w = exp_w(np.arange(X), t1, X)
    y_w = exp_w(np.arange(Y), t2, Y)

    weights = weight_constructor([x_w, y_w])
    sched_matrix = np.zeros((X, Y))
    r = np.random.random((X, Y))
    a = np.power(r, (1 / weights))
    flat = a.flatten()
    flat = np.sort(flat)[::-1]
    thresh = flat[int(p * X * Y)]
    result = np.where(a > thresh)
    for coord in list(zip(result[0], result[1])):
        sched_matrix[coord[0], coord[1]] = 1
    sched_matrix[0, 0] = 1

    return sched_matrix


def nmrbox_schedule(X, Y, t1, t2, p):
    x_w = exp_w(np.arange(X), t1, X)
    y_w = exp_w(np.arange(Y), t2, Y)

    weights = weight_constructor([x_w, y_w])
    sched_matrix = np.zeros((X, Y))
    r = np.random.random((X, Y))
    a = weights / r
    flat = a.flatten()
    flat = np.sort(flat)[::-1]
    thresh = flat[int(p * X * Y)]
    result = np.where(a > thresh)
    for coord in list(zip(result[0], result[1])):
        sched_matrix[coord[0], coord[1]] = 1
    sched_matrix[0, 0] = 1

    return sched_matrix