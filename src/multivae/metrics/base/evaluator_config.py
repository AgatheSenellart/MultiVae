from pydantic.dataclasses import dataclass
from pythae.models.base.base_config import BaseConfig


@dataclass
class EvaluatorConfig(BaseConfig):
    """

    Base config class for the evaluation modules.

    Args :
        batch_size (int) : The batch size to use in the evaluation.


    """

    batch_size: int = 512
