import logging
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.jnf_dcca import JNFDcca
from multivae.trainers.base.callbacks import TrainingCallback

from ..base import BaseTrainer
from .add_dcca_trainer_config import AddDccaTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AddDccaTrainer(BaseTrainer):
    """A specific trainer that handles the training of the DCCA module
    that is part of the JNFDcca model.

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
        model: JNFDcca,
        train_dataset: MultimodalBaseDataset,
        eval_dataset: Optional[MultimodalBaseDataset] = None,
        training_config: Optional[AddDccaTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)

        training_config.per_device_dcca_train_batch_size = min(
            len(train_dataset), training_config.per_device_dcca_train_batch_size
        )
        training_config.per_device_dcca_eval_batch_size = min(
            len(train_dataset), training_config.per_device_dcca_eval_batch_size
        )

        # TODO : maybe check that the chosen batch size is large enough and
        # that the chosen dcca batch size does'nt result in a large loss of data

        self.train_loader = self.get_train_dataloader_dcca(train_dataset)
        self.eval_loader = self.get_eval_dataloader_dcca(eval_dataset)
        self.training_config.learning_rate_vae = self.training_config.learning_rate
        self.training_config.learning_rate = self.training_config.learning_rate_dcca

    def get_train_dataloader_dcca(
        self, train_dataset: MultimodalBaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            train_sampler = None
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.per_device_dcca_train_batch_size,
            num_workers=self.training_config.train_dataloader_num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=len(train_dataset)
            > self.training_config.per_device_dcca_train_batch_size,
        )

    def get_eval_dataloader_dcca(
        self, eval_dataset: MultimodalBaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            eval_sampler = None
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.per_device_dcca_eval_batch_size,
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=(eval_sampler is None),
            sampler=eval_sampler,
            drop_last=len(eval_dataset)
            > self.training_config.per_device_dcca_eval_batch_size,
        )

    def prepare_train_step(self, epoch, best_train_loss, best_eval_loss):
        """
        Function to operate changes between train_steps such as resetting the optimizer and
        the best losses values.
        """

        if epoch == self.model.nb_epochs_dcca + 1:
            logger.info(
                "End the training of the DCCA module and move on to the joint VAE."
            )
            # Change the train and eval_loader and reset the optimizer

            self.train_loader = self.get_train_dataloader(self.train_dataset)
            self.eval_loader = self.get_eval_dataloader(self.eval_dataset)
            logger.info(
                f"Using train_loader with batch_size {len(self.train_dataset)// len(self.train_loader)} \n"
                + f"Using eval_loader with batch_size {len(self.eval_dataset)// len(self.eval_loader)} \n"
            )
            self.training_config.learning_rate = self.training_config.learning_rate_vae
            self.set_optimizer()
            self.set_scheduler()
            best_train_loss = 1e10
            best_eval_loss = 1e10

        elif epoch == self.model.nb_epochs_dcca + self.model.warmup + 1:
            # Just reset the optimizer
            logger.info(
                "End the training of the joint VAE and move on to learning the unimodal "
                " posteriors."
            )

            self.set_optimizer()
            self.set_scheduler()
            best_train_loss = 1e10
            best_eval_loss = 1e10

        return best_train_loss, best_eval_loss
