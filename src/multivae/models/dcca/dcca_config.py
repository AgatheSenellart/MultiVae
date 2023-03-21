from pydantic.dataclasses import dataclass


@dataclass
class DCCAConfig:
    n_modalities: int = 2
    embedding_dim: int = 20
    use_all_singular_values: bool = True
