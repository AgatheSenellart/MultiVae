"""Implements a Gaussian Mixture Sampler in the latent space of MultiVae models
for improved unconditional generation. A Gaussian Mixture is fitted
on the training embeddings.
"""

from .gaussian_mixture_config import GaussianMixtureSamplerConfig
from .gaussian_mixture_sampler import GaussianMixtureSampler

__all__ = ["GaussianMixtureSamplerConfig", "GaussianMixtureSampler"]
