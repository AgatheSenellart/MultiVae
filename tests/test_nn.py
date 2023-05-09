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
