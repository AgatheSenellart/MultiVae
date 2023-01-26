from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Parameters:
        input_dim (tuple): The dimension of the input data. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
    """

    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 10
