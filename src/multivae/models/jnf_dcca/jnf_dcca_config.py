from typing import Tuple, Union

from pydantic.dataclasses import dataclass

from ..dcca.dcca_config import DCCAConfig
from ..joint_models import BaseJointModelConfig


@dataclass
class JNFDccaConfig(BaseJointModelConfig):
    """
    This is the base config for the JNFDcca model.

    Args :
        warmup (int): The number of warmup epochs during training. Default to 10.
        use_likelihood_rescaling: To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings.
        nb_epochs_dcca (int) : The number of epochs during which to train the DCCA embeddings. Default to 30.
        embedding_dcca_dim (int) : The dimension of the DCCA embedding to use. Default to 20.
        use_all_singular_values (bool) : Wether to use all the singular values for the computation of the objective.
            Using True is more unstable. Default to False.


    """

    warmup: int = 10
    use_likelihood_rescaling: bool = False
    nb_epochs_dcca: int = 30
    embedding_dcca_dim: int = 20
    use_all_singular_values: bool = (
        False  # Using True generally leads to NaN in the loss.
    )
