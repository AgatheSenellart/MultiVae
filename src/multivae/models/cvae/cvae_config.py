from dataclasses import field
from typing import Dict, List, Literal, Optional

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
        sigma_variation (str): Can be used to specify a variation of the simple beta-VAE loss to use.
            Options are 'sigma_vae': learns the optimal decoder variance. or 'optimal_sigma_vae': computes
            an analytical approximation of the optimal sigma. If one of these option is chosen, beta must be
            set to 1 and decoder_dist must be set to "normal".
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
    decoder_dist: Literal["normal", "laplace", "bernoulli", "categorical", "bce"] = "normal"
    sigma_variation: Optional[Literal["sigma_vae", "optimal_sigma_vae"]] = None
    decoder_dist_params: dict = field(default_factory=lambda: {})
    custom_architectures: list = field(default_factory=lambda: [])
    sparse: bool = False
    log_sigma_init: Optional[float] = -2.0
    mean_over_batch: bool=False
    log_alpha_init: Optional[float] = 0.0
    lbd_ssim: float = 0

    def __post_init__(self):
        super().__post_init__()
        if self.sigma_variation is not None:
            if self.beta != 1.0:
                raise AttributeError(f"The sigma_variation {self.sigma_variation}"
                                     "can only be used with beta=1.")
            if self.decoder_dist != "normal":
                raise AttributeError(f"The sigma_variation {self.sigma_variation}"
                                     "can only be used with decoder_dist = 'normal'.")


