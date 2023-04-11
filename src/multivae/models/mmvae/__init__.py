"""
Implementation of the Variational Mixture-of-Experts Autoencoder model from the paper "Variational Mixture-of-Experts Autoencoders for
Multi-Modal Deep Generative Models"
(https://arxiv.org/abs/1911.03393)

"""

from .mmvae_config import MMVAEConfig
from .mmvae_model import MMVAE

__all__ = ["MMVAE", "MMVAEConfig"]
