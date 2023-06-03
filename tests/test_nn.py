import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig

from multivae.models.nn.mmnist import (
    Decoder_ResNet_AE_MMNIST,
    DecoderConvMMNIST,
    Encoder_ResNet_VAE_MMNIST,
    EncoderConvMMNIST,
)
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.models.nn.cub import CubTextEncoder, CubTextDecoderMLP

device = "cuda" if torch.cuda.is_available() else "cpu"


#### MMNIST configs ####
@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(3, 28, 28), latent_dim=10),
        BaseAEConfig(input_dim=(3, 28, 28), latent_dim=5),
    ]
)
def ae_mmnist_config(request):
    return request.param


@pytest.fixture()
def mmnist_like_data():
    return torch.rand(3, 3, 28, 28).to(device)


#### CIFAR configs ####
@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(3, 32, 32), latent_dim=10),
        BaseAEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
)
def ae_svhn_config(request):
    return request.param


@pytest.fixture()
def svhn_like_data():
    return torch.rand(3, 3, 32, 32).to(device)


#### CUB configs ####
@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(12, 356), latent_dim=10),
        BaseAEConfig(input_dim=(18, 14), latent_dim=5),
    ]
)
def ae_cub_config(request):
    return request.param


@pytest.fixture()
def cub_like_data(ae_cub_config):
    return dict(
        tokens=torch.randint(0, ae_cub_config.input_dim[1], (3, ae_cub_config.input_dim[0])).to(device),
        padding_mask=torch.randint(0, 1, (3, ae_cub_config.input_dim[0])).to(device).type(torch.float)
    )


class TestMMNISTNets:
    def test_forward(self, ae_mmnist_config, mmnist_like_data):
        encoder = EncoderConvMMNIST(ae_mmnist_config).to(device)
        decoder = DecoderConvMMNIST(ae_mmnist_config).to(device)

        enc_out = encoder(mmnist_like_data)

        assert enc_out.embedding.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )

        dec_out = decoder(enc_out.embedding)

        assert dec_out.reconstruction.shape == mmnist_like_data.shape

        encoder = Encoder_ResNet_VAE_MMNIST(ae_mmnist_config).to(device)
        decoder = Decoder_ResNet_AE_MMNIST(ae_mmnist_config).to(device)

        enc_out = encoder(mmnist_like_data)

        assert enc_out.embedding.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )

        dec_out = decoder(enc_out.embedding)

        assert dec_out.reconstruction.shape == mmnist_like_data.shape


class TestSVHNNets:
    def test_forward(self, ae_svhn_config, svhn_like_data):
        encoder = Encoder_VAE_SVHN(ae_svhn_config).to(device)
        decoder = Decoder_VAE_SVHN(ae_svhn_config).to(device)

        enc_out = encoder(svhn_like_data)

        assert enc_out.embedding.shape == (
            svhn_like_data.shape[0],
            ae_svhn_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            svhn_like_data.shape[0],
            ae_svhn_config.latent_dim,
        )

        dec_out = decoder(enc_out.embedding)

        assert dec_out.reconstruction.shape == svhn_like_data.shape


class TestCUBNets:
    def test_forward(self, ae_cub_config, cub_like_data):
        encoder = CubTextEncoder(
            ae_cub_config,
            max_sentence_length=ae_cub_config.input_dim[0],
            ntokens=ae_cub_config.input_dim[1],
            embed_size=512,
            ff_size=128
        ).to(device)
        decoder = CubTextDecoderMLP(ae_cub_config).to(device)

        enc_out = encoder(cub_like_data)

        assert enc_out.embedding.shape == (
            cub_like_data['tokens'].shape[0],
            ae_cub_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            cub_like_data['tokens'].shape[0],
            ae_cub_config.latent_dim,
        )

        dec_out = decoder(enc_out.embedding)

        assert dec_out.reconstruction.shape == cub_like_data["tokens"].shape + (ae_cub_config.input_dim[1],)
