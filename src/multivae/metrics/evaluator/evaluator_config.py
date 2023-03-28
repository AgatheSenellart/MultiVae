from pydantic.dataclasses import dataclass


@dataclass
class EvaluatorConfig:
    batch_size: int = 512
