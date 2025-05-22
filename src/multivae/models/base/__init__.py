"""**Abstract class**."""

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from .base_ae_model import BaseMultiVAE
from .base_config import BaseAEConfig, BaseMultiVAEConfig
from .base_model import BaseModel

__all__ = [
    "BaseMultiVAEConfig",
    "BaseAEConfig",
    "BaseMultiVAE",
    "BaseEncoder",
    "BaseDecoder",
    "ModelOutput",
    "BaseModel",
]
