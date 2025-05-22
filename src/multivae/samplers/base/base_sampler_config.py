"""Base sampler config, adapted from Pythae's Base sampler config for multimodal data."""

from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class BaseSamplerConfig(BaseConfig):
    """BaseSampler config class."""

    pass
