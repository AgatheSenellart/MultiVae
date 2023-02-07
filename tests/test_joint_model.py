import os
import numpy as np
import pytest
from torch import nn
import torch

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.joint_models import BaseJointModel
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from pythae.models.nn.default_architectures import Encoder_AE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.mnist.convnets import Encoder_Conv_AE_MNIST, Decoder_Conv_AE_MNIST
from pythae.models.base import BaseAEConfig

class Test_JointModel:
    
    @pytest.fixture
    def input_model(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config1 = BaseAEConfig(input_dim=(7,), latent_dim=10)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=10)

        encoders = dict(
            mod1 = Encoder_AE_MLP(config1),
            mod2 = Encoder_AE_MLP(config2)
        )
        
        decoders = dict(
            mod1 = Decoder_AE_MLP(config1),
            mod2 = Decoder_AE_MLP(config2)
        )
        
        data = dict(mod1 = torch.ones((3,7)), mod2=torch.ones((3,3)))
        return dict(encoders = encoders,model_config=model_config, decoders=decoders)

    def test(self, input_model):
        model = BaseJointModel(**input_model)
        
        assert type(model.encoders)==nn.ModuleDict
        assert isinstance(model.encoders['mod1'],Encoder_AE_MLP)
        assert isinstance(model.encoders['mod2'],Encoder_AE_MLP)
        assert type(model.decoders)==nn.ModuleDict
        assert isinstance(model.decoders['mod1'],Decoder_AE_MLP)
        assert isinstance(model.decoders['mod2'],Decoder_AE_MLP)
        
        assert isinstance(model.joint_encoder, MultipleHeadJointEncoder)
        assert model.latent_dim == input_model['model_config'].latent_dim
