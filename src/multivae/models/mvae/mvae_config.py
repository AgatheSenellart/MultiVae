from ..base import BaseMultiVAEConfig
from pydantic.dataclasses import dataclass

@dataclass
class MVAEConfig(BaseMultiVAEConfig):
    
    """Config class for the MVAE model from 'Multimodal Generative Models for Scalable Weakly-Supervised Learning'.
    https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html
    
    Args :
        k (int) : The number of subsets in the objective. Default to 1.
    
    """
    k:int=0
    
    