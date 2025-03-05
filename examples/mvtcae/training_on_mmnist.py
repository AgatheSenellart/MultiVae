"""Main script for training the MVTCAE on PolyMNIST"""

import torch
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.mmnist import DecoderResnetMMNIST, EncoderResnetMMNIST
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback

# Set data path and experiment path
DATA_PATH = "/home/asenella/data"
SAVING_PATH = "/home/asenella/expes/mvtcae_mmnist"

# Download data and split it
train_data = MMNISTDataset(data_path=DATA_PATH, split="train", download=True)
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

# Set up model configuration
modalities = ["m0", "m1", "m2", "m3", "m4"]
model_config = MVTCAEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=128,
    decoders_dist={m: "laplace" for m in modalities},
    beta=2.5,
    alpha=5.0 / 6.0,
)

# Set up encoders and decoders
encoders = {
    k: EncoderResnetMMNIST(
        private_latent_dim=0, shared_latent_dim=model_config.latent_dim
    )
    for k in modalities
}

decoders = {
    k: DecoderResnetMMNIST(latent_dim=model_config.latent_dim) for k in modalities
}

# Define model
model = MVTCAE(model_config, encoders=encoders, decoders=decoders)

# Set up training configuration
trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=1e-3,
    steps_predict=1,
    per_device_train_batch_size=128,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mvtcae_mmnist")

# Train the model
trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=[wandb_cb],
)
trainer.train()
