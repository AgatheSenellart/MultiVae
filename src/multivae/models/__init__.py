"""In this section, you will find all the models that are currently implemented in `multivae` library
"""

from .auto_model import AutoConfig, AutoModel
from .base import BaseMultiVAE, BaseMultiVAEConfig
from .jmvae import JMVAE, JMVAEConfig
from .jnf import JNF, JNFConfig
from .jnf_dcca import JNFDcca, JNFDccaConfig
from .mmvae import MMVAE, MMVAEConfig
from .mmvaePlus import MMVAEPlus, MMVAEPlusConfig
from .mopoe import MoPoE, MoPoEConfig
from .mvae import MVAE, MVAEConfig
from .mvtcae import MVTCAE, MVTCAEConfig
from .telbo import TELBO, TELBOConfig

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
    "MoPoEConfig",
    "MVTCAE",
    "MVTCAEConfig",
    "MMVAEPlusConfig",
    "MMVAEPlus",
]
