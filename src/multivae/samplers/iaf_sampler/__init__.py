"""Sampler fitting an Inverse Autoregressive Flow in the multimodal variational autoencoder's latent space.
This class is simply adapted from Pythae's iaf_sampler for the multimodal case.


"""

from .iaf_sampler import IAFSampler
from .iaf_sampler_config import IAFSamplerConfig

__all__ = ["IAFSampler", "IAFSamplerConfig"]
