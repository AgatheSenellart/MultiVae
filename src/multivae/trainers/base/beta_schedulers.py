"""Code adapted from https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb"""

import numpy as np
import math

def frange_cycle_linear(n_epochs=100, n_cycle=4, ratio=0.5):
    L = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = 1/(period*ratio) # linear schedule

    for c in range(n_cycle):

        v , i = 0 , 0
        while v <= 1 and (int(i+c*period) < n_epochs):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    


def frange_cycle_sigmoid(n_epochs=100, n_cycle=4, ratio=0.5):
    L = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = 1/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = 0 , 0
        while v <= 1:
            L[int(i+c*period)] = 1/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


#  function  = 1 − cos(a), where a scans from 0 to pi/2

def frange_cycle_cosine( n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = 1/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = 0 , 0
        while v <= 1:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    

def frange( step=1/30, n_epochs=100):
    L = np.ones(n_epochs)
    v , i = 0 , 0
    while v <= 1:
        L[i] = v
        v += step
        i += 1
    return L