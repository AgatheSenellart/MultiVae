"""Implementations of MultiVAE models
"""

from .auto_model import AutoConfig, AutoModel
from .base import BaseMultiVAE, BaseMultiVAEConfig
from .jmvae import JMVAE, JMVAEConfig
from .jnf import JNF, JNFConfig
from .mmvae import MMVAE, MMVAEConfig
from .telbo import TELBO, TELBOConfig
from .mvae import MVAE, MVAEConfig

__all__ = [
    "BaseMultiVAEConfig",
    "BaseMultiVAE",
    "JMVAEConfig",
    "JMVAE",
    "AutoConfig",
    "AutoModel",
    "JNF",
    "JNFConfig",
    "TELBO",
    "TELBOConfig",
    "MVAE",
    "MVAEConfig",
    "MMVAEConfig",
    "MMVAE"
]
