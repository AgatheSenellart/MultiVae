import argparse

import torch
import torch.nn as nn
from pythae.models.base.base_config import BaseAEConfig
from pythae.models.base.base_model import (
    BaseAEConfig,
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
)
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=8)
args = parser.parse_args()


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

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderImg, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),  # -> (2048)
            nn.Linear(2048, self.latent_dim),  # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.class_logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h), log_covariance=self.class_logvar(h)
        )


class DecoderImg(BaseDecoder):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """

    def __init__(self, model_config: BaseAEConfig):
        super(DecoderImg, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (3, 28, 28)
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)

        return ModelOutput(
            reconstruction=x_hat
        )  # NOTE: consider learning scale param, too


train_data = MMNISTDataset(data_path="~/scratch/data", split="train")


modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = MVTCAEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=512,
    decoders_dist={m: "laplace" for m in modalities},
    beta=2.5,
    alpha=5.0 / 6.0,
    decoder_dist_params={m: {"scale": 0.75} for m in modalities},
)


encoders = {
    k: EncoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: DecoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = MVTCAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=1e-3,
    steps_predict=1,
    per_device_train_batch_size=256,
    drop_last=True,
    seed=args.seed,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="reproducing_mvtcae")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

trainer._best_model.push_to_hf_hub(f"asenella/reproducing_mvtcae_seed_{args.seed}")


##################################################################
## Validate

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, FIDEvaluator, FIDEvaluatorConfig
from multivae.models.auto_model import AutoConfig, AutoModel


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h


def load_mmnist_classifiers(data_path="../../../data/clf", device="cuda"):
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")

data_path = trainer.training_dir

clfs = load_mmnist_classifiers()

model = trainer._best_model

coherences = CoherenceEvaluator(model, clfs, test_set, data_path).eval()

fids = FIDEvaluator(model, test_set).mvtcae_reproduce_fids()
