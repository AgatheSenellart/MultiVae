"""
Implementation of the DMVAE model from "Private-Shared Disentangled Multimodal VAE for Learning of Latent
Representations" .
"""

from .dmvae_config import DMVAEConfig
from .dmvae_model import DMVAE

__all__ = ["DMVAE", "DMVAEConfig"]
