from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Dict
from pythae.config import BaseConfig

@dataclass
class GMCConfig(BaseConfig):
    
    n_modalities: int = 2
    input_dims : Dict[str,tuple] = None
    common_dim: int = 20
    latent_dim: int = 20
    temperature: float = 1.0
    custom_architectures: list = field(default_factory=lambda: [])
