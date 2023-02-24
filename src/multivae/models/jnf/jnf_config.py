from typing import Tuple, Union

from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig


@dataclass
class JNFConfig(BaseJointModelConfig):
    """
    This is the base config for the JNF model.

    Args :
        warmup (int): The number of warmup epochs during training.
        use_default_flow (bool): If no flows are provided during the training, this variable becomes True
            and default MAF flows are used.
        use_likelihood_rescaling: To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings.

    """

    warmup: int = 10
    use_default_flows: bool = False
    use_likelihood_rescaling: bool = False
