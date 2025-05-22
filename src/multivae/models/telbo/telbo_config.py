from typing import Union

from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig


@dataclass
class TELBOConfig(BaseJointModelConfig):
    """Configuration class for the TELBO model from (arXiv:1705.10762 [cs, stat])
    "Generative Models of Visually Grounded Imagination" (Vedantam et al,2018).


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
        warmup (int): How many epochs to train the joint encoder and decoders before freezing them
            and learn the unimodal encoders. It is recommended to use half of the
            total training time for the first step. Default to 10.
        lambda_factors (dict[str,float]) : Ponderation factors for the reconstructions in the Joint Elbo.
            If None is provided but uses_likelihood_rescaling is True, we use the inverse product of
            dimensions as a rescaling factor for each modality. If None is provided and uses_likelihood_rescaling
            is False, each factor is set to one. Default to None.
        gamma_factors (dict[str,float]) : Ponderation factors for the reconstructions in the unimodal elbos.
            If None is provided but uses_likelihood_rescaling is True, we use the inverse product of
            dimensions as a rescaling factor for each modality. If None is provided and uses_likelihood_rescaling
            is False, each factor is set to one. Default to None.
        uses_likelihood_rescaling (bool) : Indicates how to set lambda or gamma factors when None are provided.
            Ignored when lambda_factors and gamma_factors are provided. Default to True.
    """

    warmup: int = 10
    lambda_factors: Union[dict, None] = None
    gamma_factors: Union[dict, None] = None
    uses_likelihood_rescaling: bool = True
