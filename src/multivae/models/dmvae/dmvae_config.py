from typing import Union

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class DMVAEConfig(BaseMultiVAEConfig):
    """Config class for the DMVAE model from "Private-Shared Disentangled Multimodal VAE for Learning of Latent
    Representations".


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
        modalities_specific_dims (dict): The latent dimensions for the private spaces.
        beta (float) : The scaling factor for the joint divergence term. Default to 1.
        modality_specific_betas (dict) : the betas for the private KL divergence terms. Default to None.

    """

    modalities_specific_dim: dict = None
    modalities_specific_betas: Union[dict, None] = None
    beta: float = 1.0
