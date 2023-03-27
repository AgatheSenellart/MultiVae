from typing import Dict, Tuple, Union

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseMultiVAEConfig(BaseConfig):
    """This is the base config for a Multi-Modal VAE model.

    Parameters:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        recon_losses (Dict[str, Union[function, str]]). The reconstruction loss to use per modality.
            Per modality, you can provide a string in ['mse','bce','l1']. If None is provided, an Mean-Square-Error (mse)
            is used for each modality. The choice of the reconstruction loss is equivalent to the choice of 
            the decoder distribution : 'mse' correspond to a gaussian decoder, 'l1' to a Laplace one, 'bce' to a 
            Bernoulli distribution.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for 
            computing the log-probability. 
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}
    """

    n_modalities: Union[int, None] = None
    latent_dim: int = 10
    input_dims: dict = None
    uses_default_encoders: bool = True
    uses_default_decoders: bool = True
    uses_likelihood_rescaling: bool = False
    recon_losses: dict = None
    decoder_dist_params: dict = None


@dataclass
class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
