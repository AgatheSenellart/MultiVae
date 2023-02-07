from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from ..base import BaseMultiVAEConfig

@dataclass
class JMVAEConfig(BaseMultiVAEConfig):
    """This is the config class for the jmvae model.

    Parameters:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
    """

    n_modalities: Union[int, None] = None
    latent_dim: int = 10
