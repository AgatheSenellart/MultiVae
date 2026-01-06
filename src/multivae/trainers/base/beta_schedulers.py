"""Code adapted from https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb"""

import numpy as np
import math


def frange_cycle_linear(n_epochs=100, n_cycle=4, ratio=0.5, min_beta=0, **kwargs):
    L = np.ones(n_epochs)
    period = n_epochs / n_cycle
    step = (1 - min_beta) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = min_beta, 0
        while v <= 1 and (int(i + c * period) < n_epochs):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def frange_cycle_sigmoid(n_epochs=100, n_cycle=4, ratio=0.5, min_beta=0, **kwargs):
    L = np.ones(n_epochs)
    period = n_epochs / n_cycle
    step = (1 - min_beta) / (period * ratio)  # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v, i = min_beta, 0
        while v <= 1:
            L[int(i + c * period)] = 1 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
            v += step
            i += 1
    return L


#  function  = 1 − cos(a), where a scans from 0 to pi/2


def frange_cycle_cosine(n_epochs=100, n_cycle=4, ratio=0.5, min_beta=0, **kwargs):
    L = np.ones(n_epochs)
    period = n_epochs / n_cycle
    step = (1 - min_beta) / (period * ratio)  # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):

        v, i = min_beta, 0
        while v <= 1:
            L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
            v += step
            i += 1
    return L


def frange(n_epochs=100, n_augment=30, min_beta=0, **kwargs):
    step = (1 - min_beta) / n_augment
    L = np.ones(n_epochs)
    v, i = min_beta, 0
    while v <= 1:
        L[i] = v
        v += step
        i += 1
    return L
