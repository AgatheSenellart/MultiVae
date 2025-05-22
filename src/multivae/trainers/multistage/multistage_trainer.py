import logging
from copy import deepcopy
from typing import List, Optional

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base.base_ae_model import BaseMultiVAE
from multivae.trainers.base.callbacks import TrainingCallback

from ..base import BaseTrainer
from .multistage_trainer_config import MultistageTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MultistageTrainer(BaseTrainer):
    """A specific trainer that handles the training of the joint VAE models.

    Args:
        model (BaseMultiVAE): A instance of :class:`~multivae.models.BaseMultiVAE` to train

        train_dataset (MultimodalBaseDataset): The training dataset of type
            :class:`~multivae.data.datasets.MultimodalBaseDataset`

        eval_dataset (MultimodalBaseDataset): The evaluation dataset of type
            :class:`~multivae.data.datasets.MultimodalBaseDataset`

        training_config (BaseTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`BaseTrainerConfig` is used. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallback]):
            A list of callbacks to use during training.
    """

    def __init__(
        self,
        model: BaseMultiVAE,
        train_dataset: MultimodalBaseDataset,
        eval_dataset: Optional[MultimodalBaseDataset] = None,
        training_config: Optional[MultistageTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
        checkpoint: str = None,
    ):
        super().__init__(
            model, train_dataset, eval_dataset, training_config, callbacks, checkpoint
        )

    def prepare_train_step(self, epoch, best_train_loss, best_eval_loss):
        """Function to operate changes between train_steps such as resetting the optimizer and
        the best losses values.

        """
        if epoch in self.model.reset_optimizer_epochs:
            logger.info(f"Epoch {epoch} : reset the optimizer and losses.")
            logger.info(
                "Keeping the best model obtained until here for the rest of training."
            )
            self.save_checkpoint(self._best_model, self.training_dir, epoch - 1)
            self.model = deepcopy(self._best_model).train()
            # Reset the optimizer
            self.set_optimizer()
            self.set_scheduler()
            best_train_loss = 1e12
            best_eval_loss = 1e12

        return best_train_loss, best_eval_loss

    def checktrainer(self, model):
        return
