"""
Implementation of the Variational Mixture-of-Experts Autoencoder model from the paper "Variational Mixture-of-Experts Autoencoders for
Multi-Modal Deep Generative Models"
(https://arxiv.org/abs/1911.03393)

"""

from .mmvaePlus_config import MMVAEPlusConfig
from .mmvaePlus_model import MMVAEPlus

__all__ = ["MMVAEPlus", "MMVAEPlusConfig"]
