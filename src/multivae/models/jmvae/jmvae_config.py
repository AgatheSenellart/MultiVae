from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig


@dataclass
class JMVAEConfig(BaseJointModelConfig):
    """
    This is the base config for the JMVAE model.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace','categorical']. For Bernoulli distribution,
            the decoder is expected to output **logits**. If None is provided, a normal distribution is used for each modality.
        alpha (float):  the parameter that controls the tradeoff between the ELBO and the
            regularization term. Default to 0.1.
        warmup (int): The number of warmup epochs during training. The JMVAE model uses annealing.
            The KL terms in the objective are weighted by a factor beta which is linearly brought to
            1 during the first warmup epochs. Default to 10.
        beta (float) : Weighing term for the regularization of the joint posterior to the prior.
            This parameter doesn't exist in the original method, it is a simple add-on. Default to 1.

    """

    alpha: float = 0.1
    warmup: int = 10
    beta: float = 1.0
