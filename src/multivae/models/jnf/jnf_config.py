from pydantic.dataclasses import dataclass
from typing import Literal

from ..joint_models import BaseJointModelConfig


@dataclass
class JNFConfig(BaseJointModelConfig):
    """
    This is the base config for the JNF model.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. For Bernoulli distribution,
            the decoder is expected to output **logits**. If None is provided, a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            >>> decoder_dist_params =  {'mod1' : {"scale" : 0.75}} >>>
        warmup (int): The number of warmup epochs during training. Default to 10.
        two_steps_training (bool): Whether to use a two-steps training or a one step training with annealing. Default to True.
        alpha (float): If using a one step training, the alpha parameter for the LJM term. Default to 0.1.
        beta (float): Weighing factor for the regularization of the joint VAE. Default to 1.
        add_reconstruction_terms (bool) : whether to use reconstruction terms in addition to the KL terms for
            training the unimodal posteriors.
        divide_by_prior (True) : Whether to divide by the prior when computing the Product of Experts for subset posteriors.
            Default to True. 

    """

    warmup: int = 10
    two_steps_training: bool = True
    alpha: float = 0.1
    beta: float = 1
    add_reconstruction_terms = False
    logits_to_std : Literal['standard', 'softplus'] = 'standard'
    divide_by_prior : bool = True
