import os

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_AE_MNIST,
)
from pythae.models.nn.default_architectures import Encoder_AE_MLP, Encoder_VAE_MLP
from torch import nn

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.joint_models import BaseJointModel
from multivae.models.nn.default_architectures import (
    Decoder_AE_MLP,
    MultipleHeadJointEncoder,
)


class Test_JointModel:
    @pytest.fixture
    def input_model(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config1 = BaseAEConfig(input_dim=(7,), latent_dim=10)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=10)

        encoders = dict(mod1=Encoder_AE_MLP(config1), mod2=Encoder_AE_MLP(config2))

        decoders = dict(mod1=Decoder_AE_MLP(config1), mod2=Decoder_AE_MLP(config2))

        data = dict(mod1=torch.ones((3, 7)), mod2=torch.ones((3, 3)))
        return dict(encoders=encoders, model_config=model_config, decoders=decoders)

    def test(self, input_model):
        model = BaseJointModel(**input_model)

        assert type(model.encoders) == nn.ModuleDict
        assert isinstance(model.encoders["mod1"], Encoder_AE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_AE_MLP)
        assert type(model.decoders) == nn.ModuleDict
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)

        assert isinstance(model.joint_encoder, MultipleHeadJointEncoder)
        assert model.latent_dim == input_model["model_config"].latent_dim

    @pytest.fixture
    def input2(self):
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(7,), mod2=(3,))
        )

        return dict(model_config=model_config)

    def test2(self, input2):
        model = BaseJointModel(**input2)

        assert type(model.encoders) == nn.ModuleDict
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_VAE_MLP)
        assert model.encoders["mod1"].input_dim == (7,)
        assert model.encoders["mod2"].input_dim == (3,)
        assert model.encoders["mod1"].latent_dim == 10
        assert model.encoders["mod2"].latent_dim == 10

        assert type(model.decoders) == nn.ModuleDict
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)

        assert isinstance(model.joint_encoder, MultipleHeadJointEncoder)
        assert model.latent_dim == input2["model_config"].latent_dim
