"""Define test architectures for the MHVAE model. """
import os
import shutil
import tempfile
from copy import deepcopy

import pytest
import torch
from pytest import fixture, mark
from torch import nn

from multivae.data import IncompleteDataset, MultimodalBaseDataset
from multivae.models.auto_model import AutoModel
from multivae.models.base import BaseDecoder, BaseEncoder, ModelOutput
from multivae.models.mhvae import MHVAE, MHVAEConfig
from multivae.models.nn.default_architectures import ModelOutput
from multivae.trainers import BaseTrainer, BaseTrainerConfig
# Architectures for testing
class my_input_encoder(BaseEncoder):
    """Define a test encoder"""

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.act_1 = nn.SiLU()

    def forward(self, x):

        x = self.conv0(x)
        x = self.act_1(x)

        return ModelOutput(embedding=x)


class bu_2(BaseEncoder):
    """Define a test bottom up block"""

    def __init__(self, inchannels, outchannels, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=inchannels,
                out_channels=outchannels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding=self.mu(h), log_covariance=self.log_var(h))


# Defininin top-down blocks and decoder


class td_2(nn.Module):
    """Define a test top down block"""

    def __init__(self, latent_dim):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(latent_dim, 2048), nn.ReLU())
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.linear(x)
        h = h.view(h.shape[0], 128, 4, 4)
        return self.convs(h)


class td_1(nn.Module):
    """Define a test top_down block"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class bu_1(nn.Module):
    """Define a test bottom up block"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class add_bu(nn.Module):
    """Define a test supplementary bottom up block"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.network(x)


class my_input_decoder(BaseDecoder):
    """Define a test_decoder"""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return ModelOutput(reconstruction=self.network(x))


# Defining prior blocks and posterior blocks


class prior_block(BaseEncoder):

    def __init__(self, n_channels, wn=False):
        super().__init__()
        if wn:
            self.mu = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            )
            self.logvar = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            )
        else:
            self.mu = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
            self.logvar = nn.Conv2d(n_channels, n_channels, 1, 1, 0)

    def forward(self, x):
        return ModelOutput(embedding=self.mu(x), log_covariance=self.logvar(x))


class posterior_block(BaseEncoder):

    def __init__(self, n_channels_before_concat, wn=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                2 * n_channels_before_concat,
                n_channels_before_concat,
                3,
                1,
                1,
                bias=True,
            ),
            nn.SiLU(),
        )
        if wn:
            self.mu = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
            )
            self.logvar = nn.utils.parametrizations.weight_norm(
                nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
            )
        else:
            self.mu = nn.Conv2d(
                n_channels_before_concat, n_channels_before_concat, 1, 1, 0
            )
            self.logvar = nn.Conv2d(
                n_channels_before_concat, n_channels_before_concat, 1, 1, 0
            )

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding=self.mu(h), log_covariance=self.logvar(h))
