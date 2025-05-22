import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from torch import nn


class Encoder_VAE_SVHN(BaseEncoder):
    """Simple convolutional encoder for SVHN."""

    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]
        self.fBase = 32

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(self.n_channels, self.fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(self.fBase, self.fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(self.fBase * 2, self.fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(self.fBase * 4, self.latent_dim, 4, 2, 0)
        self.c2 = nn.Conv2d(self.fBase * 4, self.latent_dim, 4, 2, 0)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x: torch.Tensor):
        e = self.enc(x)
        mu = self.c1(e).squeeze()
        lv = self.c2(e).squeeze()
        output = ModelOutput(embedding=mu, log_covariance=lv)
        return output


class Decoder_VAE_SVHN(BaseDecoder):
    """Simple convolutional encoder for SVHN."""

    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.fBase = 32
        self.nb_channels = args.input_dim[0]

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(self.fBase * 4, self.fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(self.fBase * 2, self.fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(self.fBase, self.nb_channels, 4, 2, 1, bias=True),
            nn.Sigmoid(),
            # Output size: 3 x 32 x 32
        )

    def forward(self, z: torch.Tensor):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        output = ModelOutput(reconstruction=out)
        return output
