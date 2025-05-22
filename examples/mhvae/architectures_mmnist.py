"""Defining all necessary architectures for training the MHVAE on the MMNIST dataset"""

import torch
from torch import nn

from multivae.models.base import BaseDecoder, BaseEncoder, ModelOutput


# Defining encoder and bottom-up blocks
def soft_clamp(x: torch.Tensor, v: int = 10):
    return x.div(v).tanh_().mul(v)


class my_input_encoder(BaseEncoder):
    """Custom encoder"""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.network(x)

        return ModelOutput(embedding=x)


bu_1 = nn.Sequential(
    nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True
    ),
    nn.SiLU(),
)


class bu_2(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
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
        return ModelOutput(
            embedding=soft_clamp(self.mu(h)), log_covariance=soft_clamp(self.log_var(h))
        )


# Defininin top-down blocks and decoder


class td_2(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, 2048), nn.ReLU()
        )
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.linear(x)
        h = h.view(h.shape[0], 128, 4, 4)
        return self.convs(h)


td_1 = nn.Sequential(
    nn.ConvTranspose2d(
        64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
    ),
    nn.SiLU(),
)


class my_input_decoder(BaseDecoder):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 3, 1, 1, output_padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return ModelOutput(reconstruction=self.network(z))


# Defining prior blocks and posterior blocks


class prior_block(BaseEncoder):
    def __init__(self, n_channels):
        super().__init__()

        self.mu = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
        self.logvar = nn.Conv2d(n_channels, n_channels, 1, 1, 0)

    def forward(self, x):
        return ModelOutput(
            embedding=soft_clamp(self.mu(x)), log_covariance=soft_clamp(self.logvar(x))
        )


class posterior_block(BaseEncoder):
    def __init__(self, n_channels_before_concat):
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

        self.mu = nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
        self.logvar = nn.Conv2d(
            n_channels_before_concat, n_channels_before_concat, 1, 1, 0
        )

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(
            embedding=soft_clamp(self.mu(h)), log_covariance=soft_clamp(self.logvar(h))
        )
