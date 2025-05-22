"""Tests for the BaseMultiVAE and BaseMultiVAEConfig class."""

import os

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_AE_MNIST,
)
from torch import nn

from multivae.data.datasets import IncompleteDataset
from multivae.models import AutoConfig, AutoModel
from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class TestBaseMultiVAEConfig:
    """Test the BaseMultiVAEConfig class."""

    @pytest.fixture
    def config_arguments(self):
        return dict(
            n_modalities=2,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,)),
            decoders_dist=dict(mod1="laplace", mod2="laplace"),
            decoder_dist_params=dict(mod1={"scale": 0.75}, mod2={"scale": 0.75}),
            uses_likelihood_rescaling=True,
            rescale_factors=dict(mod1=0.5, mod2=0.75),
        )

    def test_create_config(self, config_arguments):
        """Check the arguments are correctly set in the config."""
        config = BaseMultiVAEConfig(**config_arguments)

        assert config.n_modalities == config_arguments["n_modalities"]
        assert config.latent_dim == config_arguments["latent_dim"]
        assert config.decoders_dist == config_arguments["decoders_dist"]
        assert config.decoder_dist_params == config_arguments["decoder_dist_params"]
        assert (
            config.uses_likelihood_rescaling
            == config_arguments["uses_likelihood_rescaling"]
        )
        assert config.rescale_factors == config_arguments["rescale_factors"]


class Test_BaseMultiVAE:
    """Test the BaseMultiVAE class."""

    @pytest.fixture
    def input_model1(self):
        """Fixture to create a dummy model with two modalities and custom architectures."""
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)
        encoders = dict(
            mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_AE_MNIST(config)
        )
        decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))
        return dict(model_config=model_config, encoders=encoders, decoders=decoders)

    @pytest.fixture
    def input_model2(self):
        """Fixture to create a dummy model with two modalities and **no** custom architectures.
        The default ones will be used.
        """
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(1, 2), mod2=(3, 4, 4))
        )

        return dict(model_config=model_config)

    def test1(self, input_model1):
        """Test model creation with a first set of inputs.
        We verify the type of the attibutes after initialization and the coherence
        between model attributes and the model configuration.
        """
        model = BaseMultiVAE(**input_model1)

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert isinstance(model.encoders["mod2"], Encoder_Conv_AE_MNIST)
        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert isinstance(model.decoders["mod2"], Decoder_Conv_AE_MNIST)
        assert model.latent_dim == input_model1["model_config"].latent_dim
        assert model.n_modalities == input_model1["model_config"].n_modalities

    def test2(self, input_model2):
        """Test model creation with a second set of inputs.(no custom architectures)
        The default ones will be used.
        We verify the type of the attibutes after initialization and the coherence
        between model attributes and the model configuration.
        """
        model = BaseMultiVAE(**input_model2)
        # Check if the encoders are the default ones
        # and if the input dimensions are correct
        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.encoders["mod1"], Encoder_VAE_MLP)
        assert model.encoders["mod1"].input_dim == (1, 2)
        assert isinstance(model.encoders["mod2"], Encoder_VAE_MLP)
        assert model.encoders["mod2"].input_dim == (3, 4, 4)
        # Check if the decoders are the default ones
        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.decoders["mod1"], Decoder_AE_MLP)
        assert model.decoders["mod1"].input_dim == (1, 2)
        assert isinstance(model.decoders["mod2"], Decoder_AE_MLP)
        assert model.decoders["mod2"].input_dim == (3, 4, 4)
        # Check if the latent dimension is correct
        assert model.latent_dim == input_model2["model_config"].latent_dim
        assert model.n_modalities == input_model2["model_config"].n_modalities

    def test_raise_missing_input_dim(self, input_model1):
        """If not all architectures are provided (missing encoders or decoders), the user should provide
        the input dimensions of all the modalities.
        If not, an AttributeError should be raised.
        """
        ## Test incomplete input_dims when neither encoders or decoders are provided
        # Incomplete input_dims
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod2=(3, 4, 4))
        )
        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config)
        # No input_dims
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=None
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config)

        ## Test incomplete input_dims when no decoders are provided
        # Incomplete input_dims
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod2=(3, 4, 4))
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=input_model1["encoders"])
        # No input_dims
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=None
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=input_model1["encoders"])

        ## When using the 'uses_likelihood_rescaling' flag, the input_dims must be provided as well
        model_config = BaseMultiVAEConfig(
            n_modalities=2,
            latent_dim=10,
            input_dims=None,
            uses_likelihood_rescaling=True,
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(
                model_config,
                encoders=input_model1["encoders"],
                decoders=input_model1["decoders"],
            )

    def test_raises_wrong_encoders(self, input_model1):
        """If the encoders/decoders are not instances of BaseEncoder/BaseDecoder,
        an AttributeError should be raised.
        """
        # create dummy model
        model = BaseMultiVAE(**input_model1)

        encoders = dict(mod1=ModelOutput(), mod2=ModelOutput())

        with pytest.raises(AttributeError):
            model.set_encoders(encoders)

        decoders = dict(mod1=ModelOutput(), mod2=ModelOutput())

        with pytest.raises(AttributeError):
            model.set_decoders(decoders)

    def test_raises_key_error(self):
        """When both encoders and input_dims are provided,
        the encoders names must match the input_dims names.
        """
        model_config = BaseMultiVAEConfig(
            n_modalities=1, latent_dim=10, input_dims=dict(wrong_names=(3, 4, 4))
        )

        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)

        # test wrong encoder name
        encoders = dict(mod1=Encoder_VAE_MLP(config))
        decoders = dict(mod1=Decoder_Conv_AE_MNIST(config))

        with pytest.raises(KeyError):
            BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

    def test_decoders_distributions_functions(self):
        """For different decoder distributions, we test that the
        recon_log_probs functions work as expected.
        """
        # create dummy model
        model_config = BaseMultiVAEConfig(
            n_modalities=4,
            latent_dim=10,
            input_dims=dict(mod1=(3,), mod2=(3, 4), mod3=(3, 4, 4), mod4=(3, 4, 4, 4)),
            decoders_dist=dict(
                mod1="normal", mod2="bernoulli", mod3="laplace", mod4="categorical"
            ),
            decoder_dist_params=dict(
                mod1=dict(scale=12),
                mod2=None,
                mod3=dict(scale=31),
            ),
        )

        model = BaseMultiVAE(model_config)

        # Create dummy inputs
        dumb_x1 = torch.randn(2, 3)
        dumb_x2 = torch.rand(2, 3, 4)
        dumb_x2_target = torch.randint(0, 2, (2, 3, 4)).float()
        dumb_x3 = torch.randn(2, 3, 4, 4)
        dumb_x4 = torch.randint(0, 2, (2, 3, 4, 4, 4)).float()
        # Test the recon_log_probs functions
        # Check the shape of the outputs
        assert model.recon_log_probs["mod1"](dumb_x1, dumb_x1).shape == dumb_x1.shape
        assert (
            model.recon_log_probs["mod2"](dumb_x2, dumb_x2_target).shape
            == dumb_x2.shape
        )
        assert model.recon_log_probs["mod3"](dumb_x3, dumb_x3).shape == dumb_x3.shape
        assert model.recon_log_probs["mod4"](dumb_x4, dumb_x4).shape == dumb_x4.shape

    def test_raises_sanity_check_flags(self):
        """Test that an attribute error is raised when there is a
        mismatch between the encoders/decoders or the n_modalities flag.
        """
        # n_modalities = 2 but only one encoder/decoder is provided
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(3, 4, 4), mod2=(3, 4))
        )

        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)

        encoders = dict(mod1=Encoder_VAE_MLP(config))

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=encoders)

        decoders = dict(mod1=Decoder_Conv_AE_MNIST(config))

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, decoders=decoders)

        # the encoders/decoders keys don't match
        encoders = dict(mod_1=Encoder_VAE_MLP(config), mod_2=Encoder_VAE_MLP(config))

        decoders = dict(
            wrong_name_1=Decoder_Conv_AE_MNIST(config),
            wrong_name_2=Decoder_Conv_AE_MNIST(config),
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

    def test_raises_encode_error(self, input_model1):
        """Test that an error is raised when the encode function is called
        conditioning on a modality that is not present in the input data.
        """
        model = BaseMultiVAE(**input_model1)

        # Create dummy inputs with the modality mod1 missing
        inputs = IncompleteDataset(
            data=dict(mod1=torch.randn(3, 10, 2), mod2=torch.rand(3, 10, 2)),
            # The mask is zeros for mod1 since it is missing
            masks=dict(mod1=torch.zeros((3,)), mod2=torch.ones((3,))),
        )

        with pytest.raises(AttributeError):
            model.encode(inputs, cond_mod="mod1")

    def test_decode_one_latent(self, input_model1):
        """Test the basic decode function with one latent space.
        We verify that the output shape is correct.
        """
        model = BaseMultiVAE(**input_model1)

        out = ModelOutput(
            z=torch.randn(3, input_model1["model_config"].latent_dim),
            one_latent_space=True,
        )
        output = model.decode(out, modalities=["mod1"])

        assert tuple(output.mod1.shape) == (3, 10, 2)

    def test_raise_error_decode_with_wrong_input(self, input_model1):
        """Test that an error is raised when the decode function is called
        with a wrong input (not a ModelOutput).
        """
        model = BaseMultiVAE(**input_model1)

        with pytest.raises(ValueError):
            model.decode(None, modalities=["mod1"])

    def test_decode_several_latent(self):
        """Test the decode function with several latent spaces.
        We verify that the output shape is correct.
        """
        # dimension of the additional modality specific space
        mod_latent = np.random.randint(1, 100)

        # create dummy model
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        encoder_config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)
        encoders = dict(
            mod1=Encoder_VAE_MLP(encoder_config),
            mod2=Encoder_Conv_AE_MNIST(encoder_config),
        )
        decoder_config = BaseAEConfig(
            input_dim=(10, 2), latent_dim=mod_latent + model_config.latent_dim
        )
        decoders = dict(
            mod1=Decoder_AE_MLP(decoder_config),
            mod2=Decoder_Conv_AE_MNIST(decoder_config),
        )

        model = BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

        # Create dummy inputs for the decode function
        out = ModelOutput(
            z=torch.randn(3, model_config.latent_dim),
            one_latent_space=False,
            modalities_z=dict(mod1=torch.randn(3, mod_latent)),
        )

        output = model.decode(out, modalities="mod1")
        # check the shape
        assert tuple(output.mod1.shape) == (3, 10, 2)

    def test_raises_fwd_not_implemented(self, input_model1):
        """Test that an error is raised when the forward function is called
        but the forward method is not implemented in subclass.
        """
        # create dummy model
        model = BaseMultiVAE(**input_model1)

        with pytest.raises(NotImplementedError):
            model(None)

    def test_raises_nll_not_implemented(self, input_model1):
        """Test that an error is raised when the compute_joint_nll function is called
        but the compute_joint_nll method is not implemented in subclass.
        """
        model = BaseMultiVAE(**input_model1)

        with pytest.raises(NotImplementedError):
            model.compute_joint_nll(None)

    def test_generate_from_prior(self, input_model1):
        """Test the generate_from_prior function.
        We verify that the output shape is correct.
        """
        model = BaseMultiVAE(**input_model1)
        output = model.generate_from_prior(11)
        assert output.z.shape == (11, input_model1["model_config"].latent_dim)

    def test_dummy_model_saving(self, input_model1, tmp_path):
        """Test the save function."""
        # Create a dummy path and dummy model with custom architectures
        path = tmp_path / "model_save"
        model = BaseMultiVAE(**input_model1)

        assert not os.path.exists(path)
        model.save(path)
        # Check that the path exists and the files are saved
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "model_config.json"))
        assert os.path.exists(os.path.join(path, "model.pt"))
        assert os.path.exists(os.path.join(path, "encoders.pkl"))
        assert os.path.exists(os.path.join(path, "decoders.pkl"))

        # Check that an error is raised when the model is loaded from a wrong path
        with pytest.raises(FileNotFoundError):
            model._load_model_config_from_folder(tmp_path)

        with pytest.raises(FileNotFoundError):
            model._load_model_weights_from_folder(tmp_path)

        with pytest.raises(FileNotFoundError):
            model._load_custom_archi_from_folder(tmp_path, "encoders")

        # Check that an error is raised when the model.pt state_dict contains wrong keys
        torch.save({"wrong_key": torch.ones(2)}, os.path.join(tmp_path, "model.pt"))
        with pytest.raises(KeyError):
            model._load_model_weights_from_folder(tmp_path)


class TestIntegrateAutoConfig:
    """Test the AutoConfig class."""

    def test_autoconfig(self, tmp_path):
        """Test reloading a model configuration from a json file with AutoConfig.
        Check the reloaded configuration is the same as the original one.
        """
        model_config = BaseMultiVAEConfig(n_modalities=14, latent_dim=3)
        model_config.save_json(tmp_path, "model_config")
        reloaded_config = AutoConfig.from_json_file(
            os.path.join(tmp_path, "model_config.json")
        )

        assert model_config == reloaded_config

    def test_raises_not_handled(self, tmp_path):
        """Test that an error is raised when the configuration is not handled by AutoConfig."""
        training_config = BaseTrainerConfig()
        training_config.save_json(tmp_path, "training_config")
        with pytest.raises(NameError):
            _ = AutoConfig.from_json_file(
                os.path.join(tmp_path, "training_config.json")
            )


class TestIntegrateAutoModel:
    """Test the AutoModel class."""

    def test_build_automodel(self):
        """Test AutoModel init."""
        _ = AutoModel()

    def test_raises_not_handled(self):
        """Test that an error is raised when the model is not handled by AutoModel."""
        with pytest.raises(NameError):
            _ = AutoModel.load_from_folder(
                os.path.join(PATH, "tests_data", "wrong_model")
            )
