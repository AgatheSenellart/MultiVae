from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from ..base.base_config import BaseMultiVAEConfig

@dataclass
class BaseJointModelConfig(BaseMultiVAEConfig):
    """
    This is the base config for joint models.
    
    Args :
        use_default_joint (bool) :  A boolean encoding if the joint encoder used is the default one.


    """
    
    use_default_joint : bool = False

    
