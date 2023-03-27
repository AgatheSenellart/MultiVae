import os

import numpy as np
import pytest
from pythae.models.base import BaseAEConfig
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_AE_MNIST,
)
from multivae.models.nn.default_architectures import Encoder_VAE_MLP
from torch import nn

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP


class Test_BaseMultiVAE:
    @pytest.fixture
    def input_model1(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)
        encoders = dict(mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_AE_MNIST(config))
        decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))
        return dict(model_config=model_config, encoders=encoders, decoders=decoders)

    @pytest.fixture
    def input_model2(self):
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(1, 2), mod2=(3, 4, 4))
        )

        return dict(model_config=model_config)

    def test1(self, input_model1):
        model = BaseMultiVAE(**input_model1)

        assert type(model.encoders) == nn.ModuleDict
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_Conv_AE_MNIST)
        assert type(model.decoders) == nn.ModuleDict
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_Conv_AE_MNIST)
        assert model.latent_dim == input_model1["model_config"].latent_dim

    def test2(self, input_model2):
        model = BaseMultiVAE(**input_model2)

        assert type(model.encoders) == nn.ModuleDict
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert model.encoders["mod1"].input_dim == (1, 2)
        assert isinstance(model.encoders["mod2"], Encoder_VAE_MLP)
        assert model.encoders["mod2"].input_dim == (3, 4, 4)
        assert type(model.decoders) == nn.ModuleDict
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert model.decoders["mod1"].input_dim == (1, 2)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)
        assert model.decoders["mod2"].input_dim == (3, 4, 4)
        assert model.latent_dim == input_model2["model_config"].latent_dim
