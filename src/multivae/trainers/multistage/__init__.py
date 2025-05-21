"""Trainer for multistage training.
It is used for models that have a reset_optimizer_epochs attribute: JNF, TELBO.
"""

from .multistage_trainer import MultistageTrainer
from .multistage_trainer_config import MultistageTrainerConfig

__all__ = ["MultistageTrainerConfig", "MultistageTrainer"]
