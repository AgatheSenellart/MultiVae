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

from multivae.data.datasets import IncompleteDataset, MultimodalBaseDataset
from multivae.models import AutoConfig, AutoModel
from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class Test_BaseMultiVAE:
    @pytest.fixture
    def input_model1(self):
        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)
        encoders = dict(
            mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_AE_MNIST(config)
        )
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

    def test_raise_missing_input_dim(self, input_model1):
        # Test missing modality with
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod2=(3, 4, 4))
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config)

        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=None
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config)

        # Test missing modality with given encoders
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod2=(3, 4, 4))
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=input_model1["encoders"])

        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=None
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=input_model1["encoders"])

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
        # create dummy model
        model = BaseMultiVAE(**input_model1)

        encoders = dict(mod1=ModelOutput(), mod2=ModelOutput())

        with pytest.raises(AttributeError):
            model.set_encoders(encoders)

        decoders = dict(mod1=ModelOutput(), mod2=ModelOutput())

        with pytest.raises(AttributeError):
            model.set_decoders(decoders)

    def test_raises_key_error(self):
        model_config = BaseMultiVAEConfig(
            n_modalities=1, latent_dim=10, input_dims=dict(wrong_names=(3, 4, 4))
        )

        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)

        encoders = dict(mod1=Encoder_VAE_MLP(config))

        decoders = dict(mod1=Decoder_Conv_AE_MNIST(config))

        with pytest.raises(KeyError):
            BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

    def test_recon_dist(self):
        model_config = BaseMultiVAEConfig(
            n_modalities=3,
            latent_dim=10,
            input_dims=dict(mod1=(3,), mod2=(3, 4), mod3=(3, 4, 4)),
            decoders_dist=dict(mod1="normal", mod2="bernoulli", mod3="laplace"),
            decoder_dist_params=dict(
                mod1=dict(scale=12), mod2=None, mod3=dict(scale=31)
            ),
        )

        model = BaseMultiVAE(model_config)

        dumb_x1 = torch.randn(2, 3)
        dumb_x2 = torch.rand(2, 3, 4)
        dumb_x2_target = torch.randint(0, 2, (2, 3, 4)).float()
        dumb_x3 = torch.randn(2, 3, 4, 4)

        assert model.recon_log_probs["mod1"](dumb_x1, dumb_x1).shape == dumb_x1.shape
        assert (
            model.recon_log_probs["mod2"](dumb_x2, dumb_x2_target).shape
            == dumb_x2.shape
        )
        assert model.recon_log_probs["mod3"](dumb_x3, dumb_x3).shape == dumb_x3.shape

    def test_raises_sanity_check_flags(self):
        model_config = BaseMultiVAEConfig(
            n_modalities=2, latent_dim=10, input_dims=dict(mod1=(3, 4, 4), mod2=(3, 4))
        )

        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)

        encoders = dict(wrong_name=Encoder_VAE_MLP(config))

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=encoders)

        decoders = dict(wrong_name=Decoder_Conv_AE_MNIST(config))

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, decoders=decoders)

        encoders = dict(
            wrong_name1=Encoder_VAE_MLP(config), wrong_name2=Encoder_VAE_MLP(config)
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=encoders)

        decoders = dict(
            wrong_name_1=Decoder_Conv_AE_MNIST(config),
            wrong_name_2=Decoder_Conv_AE_MNIST(config),
        )

        with pytest.raises(AttributeError):
            BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

    def test_raises_encode_error(self, input_model1):
        model = BaseMultiVAE(**input_model1)

        inputs = IncompleteDataset(
            data=dict(mod1=torch.randn(3, 10, 2), mod2=torch.rand(3, 10, 2)),
            masks=dict(mod1=torch.zeros((3,)), mod2=torch.ones((3,))),
        )

        with pytest.raises(AttributeError):
            model.encode(inputs, cond_mod="mod1")

    def test_decode_one_latent(self, input_model1):
        model = BaseMultiVAE(**input_model1)

        out = ModelOutput(
            z=torch.randn(3, input_model1["model_config"].latent_dim),
            one_latent_space=True,
        )
        output = model.decode(out, modalities=["mod1"])

        assert tuple(output.mod1.shape) == (3, 10, 2)

    def test_decode_several_latent(self):
        mod_latent = np.random.randint(1, 100)

        model_config = BaseMultiVAEConfig(n_modalities=2, latent_dim=10)
        config = BaseAEConfig(input_dim=(10, 2), latent_dim=10)
        encoders = dict(
            mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_AE_MNIST(config)
        )
        config = BaseAEConfig(
            input_dim=(10, 2), latent_dim=mod_latent + config.latent_dim
        )

        decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))

        model = BaseMultiVAE(model_config, encoders=encoders, decoders=decoders)

        out = ModelOutput(
            z=torch.randn(3, model_config.latent_dim),
            one_latent_space=False,
            modalities_z=dict(mod1=torch.randn(3, mod_latent)),
        )

        output = model.decode(out, modalities="mod1")

        assert tuple(output.mod1.shape) == (3, 10, 2)

    def test_raises_fwd_not_implemented(self, input_model1):
        model = BaseMultiVAE(**input_model1)

        with pytest.raises(NotImplementedError):
            model(None)

    def test_raises_nll_not_implemented(self, input_model1):
        model = BaseMultiVAE(**input_model1)

        with pytest.raises(NotImplementedError):
            model.compute_joint_nll(None)

    def test_generate_from_prior(self, input_model1):
        model = BaseMultiVAE(**input_model1)
        output = model.generate_from_prior(11)
        assert output.z.shape == (11, input_model1["model_config"].latent_dim)

    def test_dummy_model_saving(self, input_model1, tmpdir):
        model = BaseMultiVAE(**input_model1)

        assert not os.path.exists(os.path.join(tmpdir, "model_save"))
        model.save(os.path.join(tmpdir, "model_save"))
        assert os.path.exists(os.path.join(tmpdir, "model_save"))

        with pytest.raises(FileNotFoundError):
            model._load_model_config_from_folder(tmpdir)

        with pytest.raises(FileNotFoundError):
            model._load_model_weights_from_folder(tmpdir)

        torch.save({"wrong_key": torch.ones(2)}, os.path.join(tmpdir, "model.pt"))
        with pytest.raises(KeyError):
            model._load_model_weights_from_folder(tmpdir)

        with pytest.raises(FileNotFoundError):
            model._load_custom_archi_from_folder(tmpdir, "encoders")


class TestIntegrateAutoConfig:
    def test_autoconfig(self, tmpdir):
        model_config = BaseMultiVAEConfig(n_modalities=14, latent_dim=3)
        model_config.save_json(tmpdir, "model_config")
        reloaded_config = AutoConfig.from_json_file(
            os.path.join(tmpdir, "model_config.json")
        )

        assert model_config == reloaded_config

    def test_raises_not_handled(self, tmpdir):
        training_config = BaseTrainerConfig()
        training_config.save_json(tmpdir, "training_config")
        with pytest.raises(NameError):
            _ = AutoConfig.from_json_file(os.path.join(tmpdir, "training_config.json"))


class TestIntegrateAutoModel:
    def test_build_automodel(self):
        _ = AutoModel()

    def test_raises_not_handled(self):
        with pytest.raises(NameError):
            _ = AutoModel.load_from_folder(
                os.path.join(PATH, "tests_data", "wrong_model")
            )
