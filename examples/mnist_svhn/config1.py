import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import DataLoader, random_split

from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.utils import save_all_images
from multivae.data.utils import set_inputs_to_device
from multivae.models.base import BaseMultiVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MnistSvhn(split="test", data_multiplication=5)
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

base_model_config = BaseMultiVAEConfig(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    uses_likelihood_rescaling=True,
    decoders_dist=dict(mnist="laplace", svhn="laplace"),
).to_dict() 