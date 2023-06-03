import logging
import os

import hostlist
import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets import MnistSvhn
from multivae.models import MVAE, MVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


train_data = MnistSvhn(split="test")
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

model_config = MVAEConfig(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    use_likelihood_rescaling=True,
    k=0,
)

encoders = dict(
    mnist=Encoder_VAE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Encoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

decoders = dict(
    mnist=Decoder_AE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Decoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

model = MVAE(model_config, encoders, decoders)

gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")

trainer_config = BaseTrainerConfig(
    num_epochs=60,
    output_dir="mvae_example",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=1e-3,
    steps_saving=None,
    steps_predict=1,
    no_cuda=False,
    world_size=int(os.environ["SLURM_NTASKS"]),
    dist_backend="nccl",
    rank=int(os.environ["SLURM_PROCID"]),
    local_rank=int(os.environ["SLURM_LOCALID"]),
    master_addr=hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0],
    master_port=str(12345 + int(min(gpu_ids))),
)

if int(os.environ["SLURM_PROCID"]) == 0:
    logger.info(model)
    logger.info(f"Training config: {trainer_config}\n")

callbacks = [TrainingCallback(), ProgressBarCallback()]

if trainer_config.rank == 0 or trainer_config.rank == -1:
    # Set up callbacks
    wandb_cb = WandbCallback()
    wandb_cb.setup(trainer_config, model_config=model_config, project_name="package")

    callbacks.extend([wandb_cb])

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()
