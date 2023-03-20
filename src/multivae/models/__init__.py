"""Implementations of MultiVAE models
"""

from .auto_model import AutoConfig, AutoModel
from .base import BaseMultiVAE, BaseMultiVAEConfig
from .jmvae import JMVAE, JMVAEConfig
from .jnf import JNF, JNFConfig
from .jnf_dcca import JNFDcca, JNFDccaConfig
from .mmvae import MMVAE, MMVAEConfig
from .mvae import MVAE, MVAEConfig
from .telbo import TELBO, TELBOConfig
from .mopoe import MoPoEConfig, MoPoE

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
    "MMVAE",
    "JNFDcca",
    "JNFDccaConfig",
    "MoPoE",
    "MoPoEConfig"
]
