from typing import List, Optional

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base.base_model import BaseMultiVAE
from multivae.trainers.base.callbacks import TrainingCallback

from ..base import BaseTrainer
from .jnf_trainer_config import TwoStepsTrainerConfig


class TwoStepsTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseMultiVAE,
        train_dataset: MultimodalBaseDataset,
        eval_dataset: Optional[MultimodalBaseDataset] = None,
        training_config: Optional[TwoStepsTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)

    def prepare_train_step(self, epoch, best_train_loss, best_eval_loss):
        """
        Function to operate changes between train_steps such as resetting the optimizer and
        the best losses values.
        """
        if epoch in self.model.reset_optimizer_epochs:
            # Reset the optimizer
            self.set_optimizer()
            self.set_scheduler()
            best_train_loss = 1e10
            best_eval_loss = 1e10
        return best_train_loss, best_eval_loss
