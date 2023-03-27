from pydantic.dataclasses import dataclass


@dataclass
class CoherenceEvaluatorConfig:
    batch_size: int = 512
