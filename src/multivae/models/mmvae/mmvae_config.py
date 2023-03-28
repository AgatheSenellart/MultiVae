from typing import Literal

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MMVAEConfig(BaseMultiVAEConfig):

    """
    This class is the configuration class for the MMVAE model, from
    (Variational Mixture-of-Experts Autoencoders
    for Multi-Modal Deep Generative Models, Shi et al 2019,
    https://proceedings.neurips.cc/paper/2019/hash/0ae775a8cb3b499ad1fca944e6f5c836-Abstract.html)


    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use 
            likelihood rescaling. (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        recon_losses (Dict[str, Union[function, str]]). The reconstruction loss to use per modality.
            Per modality, you can provide a string in ['mse','bce','l1']. If None is provided, an Mean-Square-Error (mse)
            is used for each modality.
        K (int) : the number of samples to use in the DreG loss. Default to 1.
        use_likelihood_rescaling (bool) : Use likelihood rescaling to mitigate modality collapse.
            Default to True.
        prior_and_posterior_dist (str) : The type of distribution to use for posterior and prior.
            Possible values ['laplace_with_softmax','normal'].
            Default to 'laplace_with_softmax' the posterior distribution that is used in
            the original paper.
        learn_prior (bool) : If True, the mean and variance of the prior are optimized during the
            training. Default to True.
    """

    K: int = 10
    prior_and_posterior_dist: Literal[
        "laplace_with_softmax", "normal"
    ] = "laplace_with_softmax"
    learn_prior: bool = True
