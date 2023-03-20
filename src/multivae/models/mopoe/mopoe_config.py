from pydantic.dataclasses import dataclass
from ..base.base_config import BaseMultiVAEConfig

@dataclass
class MoPoEConfig(BaseMultiVAEConfig):
    """
    
    subsets (dict) : Dictionary containing the subsets to consider. If None is provided, all subsets 
        are considered.
        Default to None.
    beta (float) : The weight to the KL divergence term. Default to 1.0
    """
    
    subsets : dict = None
    beta : float = 2.5 