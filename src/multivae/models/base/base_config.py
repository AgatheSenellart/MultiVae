from typing import Dict, Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Parameters:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple)
        uses_likelihood_rescaling: To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here.
        recon_losses (Dict[str, Union[function, str]]). The reconstruction loss to use per modality.
            Per modality, you can provide a string in ['mse','bce','l1'].
    """

    n_modalities: Union[int, None] = None
    latent_dim: int = 10
    input_dims: dict = None
    uses_default_encoders: bool = False
    uses_default_decoders: bool = False
    uses_likelihood_rescaling: bool = False
    recon_losses: dict = None


@dataclass
class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
