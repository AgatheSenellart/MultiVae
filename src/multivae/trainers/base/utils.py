import random

import numpy as np
import torch


def set_seed(seed: int):
    """Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``.

    Args:
        seed (int): The seed to be applied
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_dict(dict1, dict2):
    """Modify in place the first dict by adding values of the second dict."""
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k] += dict2[k]
        else:
            dict1[k] = dict2[k]
