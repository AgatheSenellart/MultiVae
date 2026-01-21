import numpy as np

def one_frange(n_epochs=100, start_augment=30,n_augment=30, max_value=100, **kwargs):
    L = np.zeros(n_epochs)
    step = max_value / n_augment
    v, i = 0, start_augment - 1
    while v <= max_value:
        L[i] = v
        v += step
        i += 1
    L[i:] = max_value
    return L

def constant(n_epochs=100, max_value=100, **kwargs):
    L = np.ones(n_epochs)*max_value
    return L