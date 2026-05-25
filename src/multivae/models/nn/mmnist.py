"""Architectures for the PolyMNIST dataset."""

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from multivae.models.base.base_config import BaseAEConfig

from .base_architectures import BaseDecoder, BaseEncoder


class Flatten(torch.nn.Module):
    """Simple transform to flatten."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    """Simple transform to reverse flatten."""

    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)


###############################################################################
### Convolutional architectures
###############################################################################


class EncoderConvMMNIST(BaseEncoder):
    """Convolutional encoder for the PolyMNIST dataset.

    Adapted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html

    """

    def __init__(self, model_config: BaseAEConfig, bias=False):
        super(EncoderConvMMNIST, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),  # -> (256, 2, 2)
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, self.latent_dim),  # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(self.latent_dim, self.latent_dim, bias=bias)
        self.class_logvar = nn.Linear(self.latent_dim, self.latent_dim, bias=bias)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h), log_covariance=self.class_logvar(h)
        )


class EncoderConvMMNIST_adapted(BaseEncoder):
    """Simple convolutional encoder with no linear layers at the end."""

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST_adapted, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.style_dim = 0
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Conv2d(128, self.latent_dim, 4, 2, 0)
        self.class_logvar = nn.Conv2d(128, self.latent_dim, 4, 2, 0)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h).squeeze(),
            log_covariance=self.class_logvar(h).squeeze(),
        )


class EncoderConvMMNIST_multilatents(BaseEncoder):
    """Adapt so that it works with multiple latent spaces models."""

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST_multilatents, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.style_dim = model_config.style_dim
        self.encoder_class = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Conv2d(128, self.latent_dim, 4, 2, 0)
        self.class_logvar = nn.Conv2d(128, self.latent_dim, 4, 2, 0)

        if self.style_dim > 0:
            self.encoder_style = nn.Sequential(  # input shape (3, 28, 28)
                nn.Conv2d(
                    3, 32, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (32, 14, 14)
                nn.ReLU(),
                nn.Conv2d(
                    32, 64, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (64, 7, 7)
                nn.ReLU(),
                nn.Conv2d(
                    64, 128, kernel_size=3, stride=2, padding=1, bias=True
                ),  # -> (128, 4, 4)
                nn.ReLU(),
            )

            self.style_mu = nn.Conv2d(128, self.style_dim, 4, 2, 0)
            self.style_logvar = nn.Conv2d(128, self.style_dim, 4, 2, 0)

    def forward(self, x):
        output = ModelOutput()
        # content branch
        h_class = self.encoder_class(x)
        output["embedding"] = self.class_mu(h_class).squeeze(-1, -2)
        output["log_covariance"] = self.class_logvar(h_class).squeeze(-1, -2)

        if self.style_dim > 0:
            # style branch
            h_style = self.encoder_style(x)
            output["style_embedding"] = self.style_mu(h_style).squeeze(-1, -2)
            output["style_log_covariance"] = self.style_logvar(h_style).squeeze(-1, -2)

        return output


class DecoderConvMMNIST(BaseDecoder):
    """Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html.
    """

    def __init__(self, model_config: BaseAEConfig):
        super(DecoderConvMMNIST, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),  # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),  # -> (128, 4, 4)
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # -> (128, 4, 4)
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
        x_hat = self.decoder(z.view(-1, z.size(-1)))
        # x_hat = torch.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *x_hat.size()[1:])
        return ModelOutput(reconstruction=x_hat)


#######################################################################################
### Resnet architectures : adapted from https://github.com/epalu/mmvaeplus
#######################################################################################


class ResnetBlock(nn.Module):
    """Resnet block for the PolyMNIST dataset.
    Adapted from https://github.com/epalu/mmvaeplus.
    """

    def __init__(
        self, nb_channels_in, nb_channels_out, nb_channels_hidden=None, bias=True
    ):
        super().__init__()
        # Attributes
        self.learn_shortcut = nb_channels_in != nb_channels_out
        if nb_channels_hidden is None:
            nb_channels_hidden = min(nb_channels_in, nb_channels_out)

        # Submodules
        self.conv_layers = nn.Sequential(
            nn.Conv2d(nb_channels_in, nb_channels_hidden, 3, stride=1, padding=1),
            nn.LeakyReLU(2e-1),
            nn.Conv2d(
                nb_channels_hidden, nb_channels_out, 3, stride=1, padding=1, bias=bias
            ),
            nn.LeakyReLU(2e-1),
        )

        if self.learn_shortcut:
            self.shortcut_layer = nn.Conv2d(
                nb_channels_in, nb_channels_out, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_layers(x)
        return x_s + 0.1 * dx

    def _shortcut(self, x):
        if self.learn_shortcut:
            return self.shortcut_layer(x)
        return x


class EncoderResnetMMNIST(BaseEncoder):
    """Resnet encoder for PolyMNIST adapted from https://github.com/epalu/mmvaeplus."""

    def __init__(self, private_latent_dim, shared_latent_dim):
        super().__init__()
        self.latent_dim = shared_latent_dim
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28
        self.multiple_latent = private_latent_dim > 0

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [ResnetBlock(nf, nf)]

        blocks_u = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_u += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        if self.multiple_latent:
            self.conv_img_w = nn.Conv2d(3, 1 * nf, 3, padding=1)
            self.resnet_w = nn.Sequential(*blocks_w)
            self.fc_mu_w = nn.Linear(self.nf0 * s0 * s0, private_latent_dim)
            self.fc_lv_w = nn.Linear(self.nf0 * s0 * s0, private_latent_dim)

        self.conv_img_u = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_u = nn.Sequential(*blocks_u)
        self.fc_mu_u = nn.Linear(self.nf0 * s0 * s0, shared_latent_dim)
        self.fc_lv_u = nn.Linear(self.nf0 * s0 * s0, shared_latent_dim)

    def forward(self, x):
        out_u = self.conv_img_u(x)
        out_u = self.resnet_u(out_u)
        out_u = out_u.view(out_u.size()[0], self.nf0 * self.s0 * self.s0)
        lv_u = self.fc_lv_u(out_u)

        output = ModelOutput(
            embedding=self.fc_mu_u(out_u),
            log_covariance=lv_u,
        )

        # batch_size = x.size(0)
        if self.multiple_latent:
            out_w = self.conv_img_w(x)
            out_w = self.resnet_w(out_w)
            out_w = out_w.view(out_w.size()[0], self.nf0 * self.s0 * self.s0)
            lv_w = self.fc_lv_w(out_w)

            output["style_embedding"] = self.fc_mu_w(out_w)
            output["style_log_covariance"] = lv_w

        return output


class DecoderResnetMMNIST(BaseDecoder):
    """Resnet decoder for PolyMNIST from https://github.com/epalu/mmvaeplus."""

    def __init__(self, latent_dim):
        """Args:
        latent_dim : total latent dimension (private + shared).
        """
        super().__init__()

        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(latent_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Sequential(
            nn.Conv2d(nf, 3, 3, padding=1), nn.LeakyReLU(2e-1)
        )

    def forward(self, z):
        out = self.fc(z).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(out)

        if len(z.size()) == 2:
            out = out.view(*z.size()[:1], *out.size()[1:])
        else:
            out = out.view(*z.size()[:2], *out.size()[1:])

        return ModelOutput(reconstruction=out)
