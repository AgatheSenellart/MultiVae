"""Architectures for the CUB dataset."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multivae.models.base import BaseDecoder, BaseEncoder, ModelOutput

#######################################################################################
############################ Text encoder and decoder #################################


class PositionalEncoding(nn.Module):
    """Taken from torch/examples."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CubTextEncoder(BaseEncoder):
    """A transformer-based encoder for text.

    Args:
        latent_dim (int): Dimension of the latent space.
        max_sentence_length (int): Maximum length of the input sentences.
        ntokens (int): Vocabulary size.
        embed_size (int): Size of the token embedding vectors. Default: 512
        nhead (int): Number of head in the MultiHeadAttention module. Default: 4
        ff_size (int): Number of units in the feedforward layers. Default: 1024
        n_layers (int): Number of Encoders layers in the TransformerEncoder. Default: 4
        dropout (float): Dropout rate. Default: 0.5
    """

    def __init__(
        self,
        latent_dim,
        max_sentence_length: int,
        ntokens: int,
        embed_size: int = 512,
        nhead: int = 4,
        ff_size: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.5,
    ):
        BaseEncoder.__init__(self)
        self.latent_dim = latent_dim

        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(ntokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, nhead, ff_size, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.mu = nn.Linear(embed_size * max_sentence_length, self.latent_dim)
        self.log_covariance = nn.Linear(
            embed_size * max_sentence_length, self.latent_dim
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.token_embedding.weight, -initrange, initrange)

    def forward(self, inputs):
        src = inputs["tokens"]
        padding_mask = inputs["padding_mask"]

        src = self.token_embedding(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(
            src, src_key_padding_mask=~padding_mask.bool()
        )
        output = ModelOutput(
            embedding=self.mu(transformer_output.reshape(src.shape[0], -1)),
            log_covariance=self.log_covariance(
                transformer_output.reshape(src.shape[0], -1)
            ),
            transformer_output=transformer_output,
        )
        return output


class CubTextDecoderMLP(BaseDecoder):
    """Simple MLP decoder for CUB text."""

    def __init__(self, args: dict):
        """A simple MLP decoder for text."""
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 512), nn.ReLU()))

        layers.append(nn.Sequential(nn.Linear(512, int(np.prod(args.input_dim)))))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor):
        output = ModelOutput()

        max_depth = self.depth
        out = z

        for i in range(max_depth):
            out = self.layers[i](out)
            if i + 1 == self.depth:
                output_shape = (*z.shape[:-1],) + self.input_dim
                output["reconstruction"] = out.reshape(output_shape)

        return output


#######################################################################################
############################ Image encoder and decoder #################################


class CUB_Resnet_Encoder(BaseEncoder):
    """Resnet Image encoder based on the one used in
    "MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises".
    (https://github.com/epalu/mmvaeplus).

    Args:
        latent_dim (int)
        s0 (int): minimum size of filtered images at the end of resnet. Must be a power of 2. Default to 16.
        nfilter (int) : Number of convolutional filters in the first layer of resnet. Default to 64.
        nfilter_max (int) : Max number of filters at the end of the resnet. Default to 1024.
    """

    def __init__(self, latent_dim, s0=16, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.latent_dim = latent_dim

        size = 64
        self.s0 = s0
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)

        self.fc_mu = nn.Linear(self.nf0 * s0 * s0, self.latent_dim)
        self.fc_logvar = nn.Linear(self.nf0 * s0 * s0, self.latent_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)

        return ModelOutput(
            embedding=self.fc_mu(actvn(out)), log_covariance=self.fc_logvar(actvn(out))
        )


class CUB_Resnet_Decoder(BaseDecoder):
    """Resnet Image Decoder based on the one used in
    "MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises".
    (https://github.com/epalu/mmvaeplus).

    Args:
        latent_dim (int)
        s0 (int): minimum size of filtered images at the end of resnet. Must be a power of 2. Default to 16.
        nfilter (int) : Number of convolutional filters in the first layer of resnet. Default to 64.
        nfilter_max (int) : Max number of filters at the end of the resnet. Default to 1024.
    """

    def __init__(self, latent_dim, s0=16, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()

        size = 64

        self.latent_dim = latent_dim

        s0 = self.s0 = s0
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(self.latent_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [ResnetBlock(nf0, nf1), nn.Upsample(scale_factor=2)]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        return ModelOutput(reconstruction=out)


class ResnetBlock(nn.Module):
    """Base residual block for resnets architectures."""

    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    """Activation function."""
    out = F.leaky_relu(x, 2e-1)
    return out
