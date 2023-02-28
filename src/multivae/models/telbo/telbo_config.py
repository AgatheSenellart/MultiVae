from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig


@dataclass
class TELBOConfig(BaseJointModelConfig):
    """The TELBO (for Triple Elbo) is a joint model which uses a two-steps training.

    Args:
        warmup (int): How many epochs to train the joint encoder and decoders.
        lambda_factors (dict[str,float]) : Ponderation factors for the reconstructions in the joint elbo.
            If None is provided but uses_likelihood_rescaling is True, we use the inverse product of
            dimensions as a rescaling factor for each modality. If None is provided and uses_likelihood_rescaling
            is False, each factor is set to one. Default to None.
        gamma_factors (dict[str,float]) : Ponderation factors for the reconstructions in the unimodal elbos.
            If None is provided but uses_likelihood_rescaling is True, we use the inverse product of
            dimensions as a rescaling factor for each modality. If None is provided and uses_likelihood_rescaling
            is False, each factor is set to one. Default to None.
        uses_likelihood_rescaling (bool) : Indicates how to set lambda or gamma factors when None are provided.
            Default to True. Ignored when lambda_factors and gamma_factors are provided.
    """

    warmup: int = 10
    lambda_factors: dict = None
    gamma_factors: dict = None
    uses_likelihood_rescaling: bool = True
