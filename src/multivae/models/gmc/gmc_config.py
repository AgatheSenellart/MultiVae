from pydantic.dataclasses import dataclass


@dataclass
class GMCConfig:
    n_modalities: int = 2
    common_dim: int = 20
    embedding_dim: int = 20
    temperature: float = 1.0
