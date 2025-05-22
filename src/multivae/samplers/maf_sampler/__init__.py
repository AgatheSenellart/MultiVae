"""Sampler fitting an Masked Autoregressive Flow in the multimodal variational autoencoder's latent space.
This class is simply adapted from Pythae's iaf_sampler for the multimodal case.


"""

from .maf_sampler import MAFSampler
from .maf_sampler_config import MAFSamplerConfig

__all__ = ["MAFSampler", "MAFSamplerConfig"]
