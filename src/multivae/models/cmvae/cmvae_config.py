from typing import Literal

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class CMVAEConfig(BaseMultiVAEConfig):
    """This class is the configuration class for the CMVAE model.

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
        K (int) : the number of samples to use in the IWAE or DreG loss. Default to 1.
        prior_and_posterior_dist (str) : The type of distribution to use for posterior and prior.
            Possible values ['laplace_with_softmax','normal_with_softplus','normal'].
            Default to 'laplace_with_softmax' the posterior distribution that is used in
            the original paper.
        learn_modality_prior (bool) : Learn modality specific priors. Should be True for the method to work.
            Default to True.
        beta (float) : Regularizes the divergence term as in beta-VAE.
            Default to 1.
        modalities_specific_dim (int) : The dimensionality of the modalitie's private latent spaces.
            Must be provided.
        reconstruction_option (Literal['single_prior','joint_prior']) : Specifies how to sample the modality specific
            variable when reconstructing/ translating modalities. Default to 'joint_prior' used in the article.
        loss (Literal['dreg_looser','iwae_looser']) : Default to 'dreg_looser'.
        number_of_clusters (int): Default to 10.
    """

    K: int = 10
    prior_and_posterior_dist: Literal[
        "laplace_with_softmax", "normal_with_softplus", "normal"
    ] = "laplace_with_softmax"
    learn_modality_prior: bool = True
    beta: float = 1.0
    modalities_specific_dim: int = None
    reconstruction_option: Literal["single_prior", "joint_prior"] = "joint_prior"
    loss: Literal["iwae_looser", "dreg_looser"] = "dreg_looser"
    number_of_clusters: int = 10
