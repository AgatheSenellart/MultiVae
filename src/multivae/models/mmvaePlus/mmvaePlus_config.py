from typing import Literal

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MMVAEPlusConfig(BaseMultiVAEConfig):
    """
    This class is the configuration class for the MMVAE+ model.


    Args:
        n_modalities (int): The number of modalities. Default: None.

        latent_dim (int): The dimension of the latent space. Default: None.

        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).

        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. Default to True.

        rescale_factors (dict[str, float]): The reconstruction rescaling factors per modality.
            If None is provided but uses_likelihood_rescaling is True, a default value proportional to the input modality
            size is computed. Default to None.

        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace','categorical'].
            For Bernoulli distribution, the decoder is expected to output **logits**. 
            If None is provided, a normal distribution is used for each modality.

        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}

        K (int) : the number of samples to use in the IWAE loss. Default to 1.

        prior_and_posterior_dist (str) : The type of distribution to use for posterior and prior.
            Possible values ['laplace_with_softmax','normal_with_softplus','normal'].
            Default to 'laplace_with_softmax' the posterior distribution that is used in
            the original paper.

        learn_shared_prior (bool) : If True, the mean and variance of the shared latent space prior are optimized during the
            training. Default to False.

        learn_modality_prior (bool) : If True, the mean and variance of the modality latent space priors are optimized during the
            training. Default to True. It is key for the method to work.

        beta (float) : When using K = 1 (ELBO loss), the beta factor regularizes the divergence term.
            Default to 1.

        modalities_specific_dim (int) : The dimensionality of the modalitie's private latent spaces.
            Must be provided.

        reconstruction_option (Literal['single_prior','joint_prior']) : Specifies how to sample the modality specific
            variable when reconstructing/ translating modalities during inference.
            Default to 'joint_prior' used in the article.

        loss (Literal['dreg_looser','iwae_looser']) : Default to 'dreg_looser'.
        
    """

    K: int = 10
    prior_and_posterior_dist: Literal[
        "laplace_with_softmax", "normal_with_softplus", "normal"
    ] = "laplace_with_softmax"
    learn_shared_prior: bool = False
    learn_modality_prior: bool = True
    beta: float = 1.0
    modalities_specific_dim: int = None
    reconstruction_option: Literal["single_prior", "joint_prior"] = "joint_prior"
    loss: Literal["iwae_looser", "dreg_looser"] = "dreg_looser"
