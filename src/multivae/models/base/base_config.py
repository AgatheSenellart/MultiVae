from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Parameters:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
    """

    n_modalities: Union[int, None] = None
    latent_dim: int = 10
    input_dims: dict = None
    uses_default_encoders: bool = True
    uses_default_decoders: bool = True
