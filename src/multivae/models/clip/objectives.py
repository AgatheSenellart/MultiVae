from typing import Dict, Union, List

import torch
from torch import Tensor
import logging
from torch.nn.functional import cross_entropy





def clip_loss( z1, z2, tau):
    
    logits = torch.mm(z1,z2.t()) * torch.exp(tau)
    labels = torch.ones(len(z1)).long().to(z1.device)
    
    l1 = cross_entropy(logits, labels)
    l2 = cross_entropy(logits.t(), labels)
    
    return (l1+l2)/2

def return_weights(m1:str, m2:str,weights:dict):
    if weights is None:
        w = 1
    elif f'{m1}_{m2}' in weights.keys():
        w = weights[f'{m1}_{m2}']
    elif f'{m2}_{m1}' in weights.keys():
        w = weights[f'{m2}_{m1}']
    else :
        w = 1
    return w
        
        
    
    