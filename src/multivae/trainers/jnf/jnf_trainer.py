from typing import List, Optional

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base.base_model import BaseMultiVAE
from multivae.trainers.base.callbacks import TrainingCallback

from ..base import BaseTrainer
from .jnf_trainer_config import JNFTrainerConfig


class JNFTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseMultiVAE,
        train_dataset: MultimodalBaseDataset,
        eval_dataset: Optional[MultimodalBaseDataset] = None,
        training_config: Optional[JNFTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)

    def train_step(self, epoch: int):
        if epoch == self.model.warmup:
            # Reset the optimizer
            self.set_optimizer()
        return super().train_step(epoch)
