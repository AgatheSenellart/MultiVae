from pydantic.dataclasses import dataclass

from ..base import BaseTrainerConfig


@dataclass
class AddDccaTrainerConfig(BaseTrainerConfig):
    """A specific trainer that handles the training of the DCCA module
    that is part of the JNFDcca model."""

    per_device_dcca_train_batch_size: int = 500
    per_device_dcca_eval_batch_size: int = 500
