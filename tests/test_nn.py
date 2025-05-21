import pytest
import torch
from pythae.models.base import BaseAEConfig

from multivae.models.nn.cub import (
    CUB_Resnet_Decoder,
    CUB_Resnet_Encoder,
    CubTextDecoderMLP,
    CubTextEncoder,
    ModelOutput,
)
from multivae.models.nn.mmnist import (
    DecoderConvMMNIST,
    DecoderResnetMMNIST,
    EncoderConvMMNIST,
    EncoderConvMMNIST_adapted,
    EncoderResnetMMNIST,
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
    """Create autoencoders configuration for PolyMNIST-like data."""
    return request.param


@pytest.fixture()
def mmnist_like_data():
    """Create a dummy PolyMNIST-like batch. (Same shape.)"""
    return torch.rand(3, 3, 28, 28).to(device)


#### SVHN configs ####
@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(3, 32, 32), latent_dim=10),
        BaseAEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
)
def ae_svhn_config(request):
    """Create autoencoders configuration for SVHN-like data"""
    return request.param


@pytest.fixture()
def svhn_like_data():
    """Create a dummy SVHN-like batch. (Same shape.)"""
    return torch.rand(3, 3, 32, 32).to(device)


#### CUB configs ####
@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(12, 356), latent_dim=10),
        BaseAEConfig(input_dim=(18, 14), latent_dim=5),
    ]
)
def ae_cub_config(request):
    """Create autoencoders configurations for CUB like data."""
    return request.param


@pytest.fixture()
def cubtext_like_data(ae_cub_config):
    """Create a dummy CUB-sentences batch for testing the architecture."""
    return dict(
        tokens=torch.randint(
            0, ae_cub_config.input_dim[1], (3, ae_cub_config.input_dim[0])
        ).to(device),
        padding_mask=torch.randint(0, 1, (3, ae_cub_config.input_dim[0]))
        .to(device)
        .type(torch.float),
    )


@pytest.fixture(
    params=[
        BaseAEConfig(input_dim=(3, 64, 64), latent_dim=10),
        BaseAEConfig(input_dim=(3, 64, 64), latent_dim=4),
    ]
)
def ae_cubimage_config(request):
    """Create autoencoders configurations for CUB like data."""
    return request.param


@pytest.fixture()
def cubimage_like_data():
    """Create a dummy CUB like batch"""
    return torch.randn((20, 3, 64, 64)).to(device)


@pytest.fixture(params=[0, 4])
def encoder_resnet_mmnist(request, ae_mmnist_config):
    """Create an EncoderResnetMMNISt for test."""
    return EncoderResnetMMNIST(
        private_latent_dim=request.param, shared_latent_dim=ae_mmnist_config.latent_dim
    ).to(device)


class TestMMNISTNets:
    """Test the forward method of PolyMNIST architectures on a dummy PolyMNIST
    batch.
    We check that the output is a ModelOutput and that the embeddings are the right shape.
    """

    def test_forward(self, ae_mmnist_config, mmnist_like_data, encoder_resnet_mmnist):
        # Test convolutional networks
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

        # Test adapted convolutional network (no linear layer)
        encoder = EncoderConvMMNIST_adapted(ae_mmnist_config).to(device)
        enc_out = encoder(mmnist_like_data)

        assert enc_out.embedding.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            mmnist_like_data.shape[0],
            ae_mmnist_config.latent_dim,
        )

        # Test Resnet networks
        encoder = encoder_resnet_mmnist
        decoder = DecoderResnetMMNIST(latent_dim=ae_mmnist_config.latent_dim).to(device)

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
    """Test the forward method of SVHN architectures on a dummy SVHN
    batch.
    We check that the output is a ModelOutput and that the embeddings are the right shape.
    """

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
    """Test the forward method of CUB architectures on a dummy CUB
    batch.
    We check that the output is a ModelOutput and that the embeddings are the right shape.
    """

    def test_text_forward(self, ae_cub_config, cubtext_like_data):
        encoder = CubTextEncoder(
            ae_cub_config.latent_dim,
            max_sentence_length=ae_cub_config.input_dim[0],
            ntokens=ae_cub_config.input_dim[1],
            embed_size=512,
            ff_size=128,
        ).to(device)
        decoder = CubTextDecoderMLP(ae_cub_config).to(device)

        enc_out = encoder(cubtext_like_data)

        assert enc_out.embedding.shape == (
            cubtext_like_data["tokens"].shape[0],
            ae_cub_config.latent_dim,
        )
        assert enc_out.log_covariance.shape == (
            cubtext_like_data["tokens"].shape[0],
            ae_cub_config.latent_dim,
        )

        dec_out = decoder(enc_out.embedding)

        assert dec_out.reconstruction.shape == cubtext_like_data["tokens"].shape + (
            ae_cub_config.input_dim[1],
        )

    def test_image_forward(self, ae_cubimage_config, cubimage_like_data):
        encoder = CUB_Resnet_Encoder(latent_dim=ae_cubimage_config.latent_dim).to(
            device
        )
        decoder = CUB_Resnet_Decoder(latent_dim=ae_cubimage_config.latent_dim).to(
            device
        )

        output = encoder(cubimage_like_data)

        assert isinstance(output, ModelOutput)
        assert output.embedding.shape == (
            cubimage_like_data.shape[0],
            ae_cubimage_config.latent_dim,
        )

        assert output.log_covariance.shape == (
            cubimage_like_data.shape[0],
            ae_cubimage_config.latent_dim,
        )

        dec_out = decoder(output.embedding)

        assert dec_out.reconstruction.shape == cubimage_like_data.shape
