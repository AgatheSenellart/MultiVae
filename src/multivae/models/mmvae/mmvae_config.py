from typing import Literal

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MMVAEConfig(BaseMultiVAEConfig):
    """This class is the configuration class for the MMVAE model, from
    (Variational Mixture-of-Experts Autoencoders
    for Multi-Modal Deep Generative Models, Shi et al 2019,
    https://proceedings.neurips.cc/paper/2019/hash/0ae775a8cb3b499ad1fca944e6f5c836-Abstract.html).


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
        K (int) : the number of samples to use for importance sampling. Default to 1.
        prior_and_posterior_dist (str) : The type of distribution to use for posterior and prior.
            Possible values ['laplace_with_softmax','normal'].
            Default to 'laplace_with_softmax' the posterior distribution that is used in
            the original paper.
        learn_prior (bool) : If True, the mean and variance of the prior are optimized during the
            training. Default to True.
        beta (float) : Regularizes the divergence term.
            Default to 1.
        loss (Literal) : Either 'iwae_looser' or 'dreg_looser'.
    """

    K: int = 10
    prior_and_posterior_dist: Literal["laplace_with_softmax", "normal"] = (
        "laplace_with_softmax"
    )
    learn_prior: bool = True
    beta: float = 1.0
    loss: Literal["iwae_looser", "dreg_looser"] = "dreg_looser"
