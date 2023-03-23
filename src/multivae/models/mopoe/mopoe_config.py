from typing import List, Union, Dict
from pydantic.dataclasses import dataclass
from ..base.base_config import BaseMultiVAEConfig

@dataclass
class MoPoEConfig(BaseMultiVAEConfig):
    """
    
    subsets (List[list] or Dict[list]) : List or dictionary containing the subsets to consider. If None is provided,
        all subsets are considered. Examples of valid inputs : [['mod_1', 'mod_2], ['mod_1'], ['mod_2']] 
        or {'s1' : ['mod_1', 'mod_2], 's2' : ['mod_1'], 's3' : ['mod_2']}
        Default to None.
    beta (float) : The weight to the KL divergence term in the ELBO. Default to 1.0
    """
    
    subsets : Union[Dict[str, list],List[list]] = None
    beta : float = 1.0
    decoder_scale : float = 0.75