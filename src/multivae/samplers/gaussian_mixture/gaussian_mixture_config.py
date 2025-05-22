from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class GaussianMixtureSamplerConfig(BaseSamplerConfig):
    """Gaussian mixture sampler config class.

    Args:
        n_components (int): The number of Gaussians in the mixture. Default to 10
    """

    n_components: int = 10
