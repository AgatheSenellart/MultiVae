from pydantic.dataclasses import dataclass


@dataclass
class CLIPConfig:
    n_modalities: int = 2
    joint_embedding_dim: int = 20
    weights : dict = None
