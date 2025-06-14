from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class IAFSamplerConfig(BaseSamplerConfig):
    """This is the IAF sampler model configuration instance.

    Args:
        n_made_blocks (int): The number of MADE model to consider in the IAF. Default: 2.
        n_hidden_in_made (int): The number of hidden layers in the MADE models. Default: 3.
        hidden_size (list): The number of unit in each hidder layer. The same number of units is
            used across the `n_hidden_in_made` and `n_made_blocks`. Default: 3.
        include_batch_norm (bool): Whether to include batch normalization after each
            :class:`~pythae.models.normalizing_flows.MADE` layers. Default: False.
    """

    n_made_blocks: int = 2
    n_hidden_in_made: int = 3
    hidden_size: int = 128
    include_batch_norm: bool = False
