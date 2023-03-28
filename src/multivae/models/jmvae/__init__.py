"""
Implementation of the Joint Multimodal VAE model from the paper "Joint Multimodal Learning with Deep
Generative Models" (http://arxiv.org/abs/1611.01891).
"""

from .jmvae_config import JMVAEConfig
from .jmvae_model import JMVAE

__all__ = ["JMVAEConfig", "JMVAE"]
