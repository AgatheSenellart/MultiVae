import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets import MnistSvhn
from multivae.models import JNFDcca, JNFDccaConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
# Define paths
DATA_PATH = "/home/asenella/data/"
SAVE_PATH = "/home/asenella/expes/jnfdcca_mnist_svhn"

# Load data
train_data = MnistSvhn(data_path=DATA_PATH,split="train", download=True)
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

# Define model config
model_config = JNFDccaConfig(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    nb_epochs_dcca=50,
    warmup=50,
    uses_likelihood_rescaling=True,
)

dcca_networks = dict(
    mnist=Encoder_VAE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Encoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

decoders = dict(
    mnist=Decoder_AE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Decoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)
# Define model
model = JNFDcca(
    model_config, encoders=None, decoders=decoders, dcca_networks=dcca_networks
)

# Define trainer config. This model uses the AddDccaTrainer
trainer_config = AddDccaTrainerConfig(
    num_epochs=150,
    learning_rate=1e-3,
    steps_predict=1,
    per_device_dcca_train_batch_size=500,
    output_dir=SAVE_PATH
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="package")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

# Train model
trainer = AddDccaTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

