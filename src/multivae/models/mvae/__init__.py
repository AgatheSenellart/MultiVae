"""
Implementation of the Multi-modal VAE model from "Multimodal Generative Models for Scalable
Weakly-Supervised Learning" (https://arxiv.org/abs/1802.05335).
"""

from .mvae_config import MVAEConfig
from .mvae_model import MVAE

__all__ = ["MVAE", "MVAEConfig"]
