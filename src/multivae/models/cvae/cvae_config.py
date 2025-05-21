from dataclasses import field
from typing import Dict, List, Literal

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class CVAEConfig(BaseConfig):
    """This is the configuration class for the Conditional Variational Autoencoder model.

    Args:
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        latent_dim (int): The dimension of the latent space. Default: 10.
        conditioning_modalities (List[str]): The modalities to condition the model on.
        main_modality (str): The main modality to reconstruct.
        beta (float): The parameter that weighs the KL divergence in the ELBO. Default to 1.0.
        decoder_dist (str): The decoder distribution to use. Possible values ['normal', 'bernoulli', 'laplace', 'categorical'].
            For Bernoulli distribution, the decoder is expected to output **logits**.
        decoder_dist_params (dict) : To eventually specify parameters for the output decoder distribution.
            Default to None.

    """

    conditioning_modalities: List[str]
    main_modality: str
    input_dims: Dict[str, tuple] = None
    latent_dim: int = 10
    beta: float = 1.0
    decoder_dist: Literal["normal", "laplace", "bernoulli", "categorical"] = "normal"
    decoder_dist_params: dict = field(default_factory=lambda: {})
    custom_architectures: list = field(default_factory=lambda: [])
