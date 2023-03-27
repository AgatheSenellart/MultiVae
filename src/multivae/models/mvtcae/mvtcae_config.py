from typing import List, Union, Dict
from pydantic.dataclasses import dataclass
from ..base.base_config import BaseMultiVAEConfig

@dataclass
class MVTCAEConfig(BaseMultiVAEConfig):
    """
    This is the base config class for the MVTCAE model from 
    'Multi-View Representation Learning via Total Correlation Objective' Neurips 2021. 
    The code is based on the original implementation that can be found here :
    https://github.com/gr8joo/MVTCAE/blob/master/run_epochs.py
    
    alpha (float) : The parameter that ponderates the total correlation ratio in the loss.  
    beta (float) : The parameter that weights the sum of all KLs
    """
    
    alpha : float = 0.1
    beta: float = 2.5
    