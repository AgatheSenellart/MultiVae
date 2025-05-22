from pydantic.dataclasses import dataclass

from ..base.base_config import BaseMultiVAEConfig


@dataclass
class BaseJointModelConfig(BaseMultiVAEConfig):
    """This is the base config for joint models."""

    pass
