from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from ..base.base_config import BaseMultiVAEConfig

@dataclass
class JMVAEConfig(BaseMultiVAEConfig):
    """This is the base config for the JMVAE model.

    """
    
    alpha : float = 0.1

    
