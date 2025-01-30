"""
Implementation of "MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises"
(https://openreview.net/forum?id=sdQGxouELX)

"""

from .mmvaePlus_config import MMVAEPlusConfig
from .mmvaePlus_model import MMVAEPlus

__all__ = ["MMVAEPlus", "MMVAEPlusConfig"]
