import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import DataLoader, random_split

from multivae.data.datasets import MMNISTDataset
from multivae.data.datasets.utils import save_all_images
from multivae.data.utils import set_inputs_to_device
from multivae.models import MoPoE,MoPoEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

import torch
import torch.nn as nn
from pythae.models.base.base_model import BaseDecoder, BaseEncoder, BaseAEConfig, ModelOutput

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)

class EncoderImg(BaseEncoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, model_config:BaseAEConfig):
        super(EncoderImg, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048,self.latent_dim ),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.class_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(embedding = self.class_mu(h),
                           log_covariance = self.class_logvar(h))


class DecoderImg(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, model_config : BaseAEConfig):
        super(DecoderImg, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, z):

        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        
        return ModelOutput(reconstruction = x_hat)  # NOTE: consider learning scale param, too



train_data = MMNISTDataset(data_path = "../../../data/MMNIST",split="train")
train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)
print(len(train_data), len(eval_data))

modalities = ['m0','m1','m2', 'm3','m4']

model_config = MoPoEConfig(
    n_modalities=5,
    input_dims={k : (3,28,28) for k in modalities},
    latent_dim=512,
    recon_losses={m : 'l1' for m in modalities },
    decoder_scale=0.75,
    beta=2.5 # The std deviation of decoder in original implementation is 0.75
    
)



encoders = { k : EncoderImg(BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))) for k in modalities}

decoders = {
    k :DecoderImg(BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))) for k in modalities
}

model = MoPoE(
    model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=0.5e-4,
    steps_predict=1,
    per_device_train_batch_size=256,
    drop_last=True,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="reproduce_mopoe")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks
)
trainer.train()

# data = set_inputs_to_device(eval_data[:100], device="cuda")
# nll = model.compute_joint_nll(data)
# print(nll)
