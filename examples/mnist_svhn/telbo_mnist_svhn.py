import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets import MnistSvhn
from multivae.models import TELBO, TELBOConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MnistSvhn(split="test")
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

model_config = TELBOConfig(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    warmup=30,
    gamma_factors=dict(mnist=1.0, svhn=1.0),
)

encoders = dict(
    mnist=Encoder_VAE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Encoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

decoders = dict(
    mnist=Decoder_AE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Decoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

model = TELBO(model_config, encoders, decoders)

trainer_config = TwoStepsTrainerConfig(
    num_epochs=60, learning_rate=1e-3, steps_predict=1
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="package")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

# data = set_inputs_to_device(eval_data[:100], device="cuda")
# nll = model.compute_joint_nll(data)
# print(nll)
