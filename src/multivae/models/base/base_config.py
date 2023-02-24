from typing import Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Parameters:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple)
    """

    n_modalities: Union[int, None] = None
    latent_dim: int = 10
    input_dims: dict = None
    uses_default_encoders: bool = False
    uses_default_decoders: bool = False


@dataclass
class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
