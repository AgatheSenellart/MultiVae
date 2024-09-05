from pydantic.dataclasses import dataclass
from typing import Dict


@dataclass
class GMCConfig:
    n_modalities: int = 2
    input_dims : Dict[str,tuple] = None
    common_dim: int = 20
    latent_dim: int = 20
    temperature: float = 1.0
