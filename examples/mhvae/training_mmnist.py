"""Main code for training a MHVAE model on the PolyMNIST dataset"""

import argparse

from architectures_mmnist import (
    bu_1,
    bu_2,
    my_input_decoder,
    my_input_encoder,
    posterior_block,
    prior_block,
    td_1,
    td_2,
)
from torch.utils.data import random_split

from multivae.data.datasets import MMNISTDataset
from multivae.models.mhvae import MHVAE, MHVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback

# Set data path and experiment path:
DATA_PATH = "/home/asenella/data"
SAVING_PATH = "/home/asenella/expes/mhvae_mmnist"

# Parser to define if weights are shared or not
parser = argparse.ArgumentParser()
parser.add_argument("--share_weights", action="store_true")
args = parser.parse_args()

# Define architectures and posterior blocks with or without shared weights
if args.share_weights:
    posterior_blocks = [posterior_block(32), posterior_block(64)]
else:
    posterior_blocks = {
        f"m{i}": [posterior_block(32), posterior_block(64)] for i in range(5)
    }

# Define model configuration
model_config = MHVAEConfig(
    n_modalities=5,
    latent_dim=128,  # For the deepest variable
    input_dims={f"m{i}": (3, 28, 28) for i in range(5)},
    n_latent=3,  # Number of hierarchical latent variables
    beta=1.0,
)

model = MHVAE(
    model_config=model_config,
    encoders={f"m{i}": my_input_encoder() for i in range(5)},
    decoders={f"m{i}": my_input_decoder() for i in range(5)},
    bottom_up_blocks={f"m{i}": [bu_1, bu_2(model_config.latent_dim)] for i in range(5)},
    top_down_blocks=[td_1, td_2(model_config.latent_dim)],
    prior_blocks=[prior_block(32), prior_block(64)],
    posterior_blocks=posterior_blocks,
)

# Import dataset
train_set = MMNISTDataset(DATA_PATH)
train_set, eval_set = random_split(train_set, [0.8, 0.2])

# Define training configuration
trainer_config = BaseTrainerConfig(
    output_dir=SAVING_PATH,
    per_device_eval_batch_size=128,
    per_device_train_batch_size=128,
    num_epochs=100,
    steps_predict=5,
    learning_rate=1e-4,
)

# Set up wandb callback (Optional )
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, "mhvae_mmnist")

# Define trainer and train
trainer = BaseTrainer(
    training_config=trainer_config,
    model=model,
    train_dataset=train_set,
    eval_dataset=eval_set,
    callbacks=[wandb_cb],
)

trainer.train()
