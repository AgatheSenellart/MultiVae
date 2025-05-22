from dataclasses import field
from typing import Dict, Literal, Optional, Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        rescale_factors (dict[str, float]): The reconstruction rescaling factors per modality.
            If None is provided but uses_likelihood_rescaling is True, a default value proportional to the input modality
            size is computed. Default to None.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. For Bernoulli distribution,
            the decoder is expected to output **logits**. If None is provided, a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary with
            :code:`decoder_dist_params =  {'mod1' : {"scale" : 0.75}}`.
    """

    n_modalities: int
    latent_dim: int = 10
    input_dims: Optional[dict] = None
    uses_likelihood_rescaling: bool = False
    rescale_factors: Optional[dict] = None
    decoders_dist: Union[
        Dict[str, Literal["normal", "bernoulli", "laplace", "categorical"]], None
    ] = None
    decoder_dist_params: Union[dict, None] = None
    custom_architectures: list = field(default_factory=lambda: [])


@dataclass
class EnvironmentConfig(BaseConfig):
    """Base environment config to save python version."""

    python_version: str = "3.8"


@dataclass
class BaseAEConfig(BaseConfig):
    """This is the base configuration instance of encoders/decoders models deriving from
    :class:`~pythae.config.BaseConfig`.

    Args:
        input_dim (tuple): The input_data dimension (channels X x_dim X y_dim)
        latent_dim (int): The latent space dimension. Default: None.
        style_dim (int) : For models with private latent spaces for each modality. Default: 0.
    """

    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 10
    style_dim: int = 0
