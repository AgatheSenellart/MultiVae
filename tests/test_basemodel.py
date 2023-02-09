import os
import numpy as np
import pytest
from torch import nn

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from pythae.models.nn.default_architectures import Encoder_AE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.mnist.convnets import Encoder_Conv_AE_MNIST, Decoder_Conv_AE_MNIST
from pythae.models.base import BaseAEConfig

class Test_BaseMultiVAE:
    
    @pytest.fixture
    def input_model(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config = BaseAEConfig(input_dim=(10,2), latent_dim=10)
        encoders = dict(
            mod1 = Encoder_AE_MLP(config),
            mod2 = Encoder_Conv_AE_MNIST(config)
        )
        decoders = dict(
            mod1 = Decoder_AE_MLP(config),
            mod2 = Decoder_Conv_AE_MNIST(config)
        )
        return dict(model_config=model_config, encoders=encoders,decoders=decoders)

    def test(self, input_model):
        model = BaseMultiVAE(**input_model)

        assert type(model.encoders)==nn.ModuleDict
        assert isinstance(model.encoders['mod1'],Encoder_AE_MLP)
        assert isinstance(model.encoders['mod2'],Encoder_Conv_AE_MNIST)
        assert type(model.decoders)==nn.ModuleDict
        assert isinstance(model.decoders['mod1'],Decoder_AE_MLP)
        assert isinstance(model.decoders['mod2'],Decoder_Conv_AE_MNIST)
        assert model.latent_dim == input_model['model_config'].latent_dim