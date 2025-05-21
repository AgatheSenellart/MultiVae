import pytest
from pythae.models.base import BaseAEConfig
from pythae.models.nn.default_architectures import Encoder_AE_MLP
from torch import nn

from multivae.models.base import BaseMultiVAEConfig
from multivae.models.joint_models import BaseJointModel
from multivae.models.nn.default_architectures import (
    Decoder_AE_MLP,
    Encoder_VAE_MLP,
    MultipleHeadJointEncoder,
)


class Test_JointModel:
    """Test BaseJointModel class."""

    @pytest.fixture
    def input_model_1(self):
        """Create custom architectures and configuration for BaseJointModel"""
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config1 = BaseAEConfig(input_dim=(7,), latent_dim=10)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=10)

        encoders = dict(mod1=Encoder_AE_MLP(config1), mod2=Encoder_AE_MLP(config2))

        decoders = dict(mod1=Decoder_AE_MLP(config1), mod2=Decoder_AE_MLP(config2))

        return dict(encoders=encoders, model_config=model_config, decoders=decoders)

    def test_init_1(self, input_model_1):
        """Test the initialization of BaseJointModel class with
        custom architectures and configuration 1.
        Check the attributes types and values.
        """
        model = BaseJointModel(**input_model_1)

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.encoders["mod1"], Encoder_AE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_AE_MLP)
        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)
        assert isinstance(model.joint_encoder, MultipleHeadJointEncoder)
        assert model.latent_dim == input_model_1["model_config"].latent_dim

    @pytest.fixture
    def input_model_2(self):
        """Create a second configuration for testing init.
        This time we test without providing custom architectures.
        The model will use the default architectures.
        """
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(7,), mod2=(3,))
        )

        return dict(model_config=model_config)

    def test_init_2(self, input_model_2):
        """Test init of BaseJointModel class without custom architectures.
        We check that the default architectures are used and that the
        attributes are properly set up.
        """
        model = BaseJointModel(**input_model_2)

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_VAE_MLP)
        assert model.encoders["mod1"].input_dim == (7,)
        assert model.encoders["mod2"].input_dim == (3,)
        assert model.encoders["mod1"].latent_dim == 10
        assert model.encoders["mod2"].latent_dim == 10

        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)
        assert isinstance(model.joint_encoder, MultipleHeadJointEncoder)
        assert model.latent_dim == input_model_2["model_config"].latent_dim
