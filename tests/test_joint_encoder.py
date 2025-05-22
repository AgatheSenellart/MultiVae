import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.nn.default_architectures import Encoder_AE_MLP
from torch import nn

from multivae.models.base import BaseMultiVAEConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder


class Test_MultipleHeadJointEncoder:
    """Test the MultipleHeadJointEncoder class.
    We check that the attributes and modules are properly set up
    during initialization.
    """

    @pytest.fixture
    def input_model(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config1 = BaseAEConfig(input_dim=(7,), latent_dim=10)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=10)

        encoders = dict(mod1=Encoder_AE_MLP(config1), mod2=Encoder_AE_MLP(config2))
        data = dict(mod1=torch.ones((3, 7)), mod2=torch.ones((3, 3)))
        return dict(dict_encoders=encoders, args=model_config, data=data)

    def test(self, input_model):
        model = MultipleHeadJointEncoder(**input_model)

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.encoders["mod1"], Encoder_AE_MLP)
        assert model.encoders["mod1"] != input_model["dict_encoders"]["mod1"]
        assert model.latent_dim == input_model["args"].latent_dim

        # forward pass
        outputs = model(input_model["data"])
        assert outputs.embedding.shape == (3, 10)
        assert outputs.log_covariance.shape == (3, 10)
