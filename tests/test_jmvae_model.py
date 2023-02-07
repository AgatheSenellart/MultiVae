import os
import numpy as np
import pytest

from multivae.data.datasets.base import MultimodalBaseDataset
import os
import numpy as np
import pytest
from torch import nn
import torch

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.jmvae import JMVAE, JMVAEConfig
from multivae.models.joint_models import BaseJointModel
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.mnist.convnets import Encoder_Conv_AE_MNIST, Decoder_Conv_AE_MNIST
from pythae.models.base import BaseAEConfig


class Test:
    @pytest.fixture
    def input1(self):
        
        # Create simple small dataset
        data = dict(
            mod1 = torch.Tensor([[1.0,2.0],[4.0,5.0]]),
            mod2 = torch.Tensor([[67.1,2.3,3.0],[1.3,2.,3.]]),
        )
        labels = np.array([0,1])
        dataset = MultimodalBaseDataset(data, labels)
        
        # Create an instance of jmvae model
        model_config = JMVAEConfig(n_modalities=2, latent_dim=5)
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)

        encoders = dict(
            mod1 = Encoder_VAE_MLP(config1),
            mod2 = Encoder_VAE_MLP(config2)
        )
        
        decoders = dict(
            mod1 = Decoder_AE_MLP(config1),
            mod2 = Decoder_AE_MLP(config2)
        )
        
        return dict(model_config = model_config,
                    encoders = encoders,
                    decoders = decoders,
                    dataset = dataset)

    def test1(self, input1):
        model = JMVAE(**input1)

        assert model.alpha == input1['model_config'].alpha
        
        loss = model(input1['dataset'], epoch=2, warmup=2).loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        
