from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from ..joint_models import BaseJointModelConfig

@dataclass
class JMVAEConfig(BaseJointModelConfig):
    """
    This is the base config for the JMVAE model. 
    
    Args :
        alpha (float) :  the parameter that controls the tradeoff between the ELBO and the regularization term. Default to 0.1.


    """
    
    alpha : float = 0.1

    
