import os

import numpy as np
import pytest
import torch
from pythae.trainers import BaseTrainerConfig

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import DMVAE, MVAE, DMVAEConfig, MVAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, ModelOutput
from multivae.samplers.iaf_sampler import IAFSampler, IAFSamplerConfig

from .encoders import EncoderTest, EncoderTestMultilatents


class Test_IAFSampler:
    """Test the IAFSampler class."""

    @pytest.fixture
    def dataset(self):
        """Dummy dataset"""
        data = dict(
            mod1=torch.randn(20, 2),
            mod2=torch.randn(20, 3),
            mod3=torch.randn(20, 4),
            mod4=torch.randn(20, 4),
        )
        labels = np.array([0] * 10 + [1] * 10)
        dataset = MultimodalBaseDataset(data, labels)

        return dataset

    @pytest.fixture(params=[True, False])
    def one_latent_space(self, request):
        """Test the IAF sampler on one or multiple latent spaces."""
        return request.param

    @pytest.fixture
    def archi_and_config(self, beta, one_latent_space):
        """Create architectures and configs for test model."""
        if one_latent_space:
            # Create an instance of mvae model
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

            encoders = dict(
                mod1=EncoderTest(config1),
                mod2=EncoderTest(config2),
                mod3=EncoderTest(config3),
                mod4=EncoderTest(config3),
            )

            model_config = MVAEConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
            )

            decoders = dict(
                mod1=Decoder_AE_MLP(config1),
                mod2=Decoder_AE_MLP(config2),
                mod3=Decoder_AE_MLP(config3),
                mod4=Decoder_AE_MLP(config3),
            )

        else:
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5, style_dim=4)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5, style_dim=2)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5, style_dim=3)

            encoders = dict(
                mod1=EncoderTestMultilatents(config1),
                mod2=EncoderTestMultilatents(config2),
                mod3=EncoderTestMultilatents(config3),
                mod4=EncoderTestMultilatents(config3),
            )
            model_config = DMVAEConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
                modalities_specific_dim=dict(mod1=4, mod2=2, mod3=3, mod4=3),
            )
            decoders = dict(
                mod1=Decoder_AE_MLP(BaseAEConfig(input_dim=(2,), latent_dim=9)),
                mod2=Decoder_AE_MLP(BaseAEConfig(input_dim=(3,), latent_dim=7)),
                mod3=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
                mod4=Decoder_AE_MLP(BaseAEConfig(input_dim=(4,), latent_dim=8)),
            )

        return dict(encoders=encoders, decoders=decoders, model_config=model_config)

    @pytest.fixture(params=[1.0, 1.5, 2.0])
    def beta(self, request):
        """Test the IAF sampler with different beta values."""
        beta = request.param

        return beta

    @pytest.fixture(params=[True, False])
    def model(self, archi_and_config, one_latent_space, request):
        """Create a MVAE or DMVAE model for testing the IAF sampler in different settings."""
        custom = request.param

        if one_latent_space:
            model_class = MVAE
        else:
            model_class = DMVAE

        if custom:
            model = model_class(**archi_and_config)
        else:
            model = model_class(archi_and_config["model_config"])
        return model

    @pytest.fixture(params=[0, 1])
    def iaf_sampler_config(self, request):
        """IAF sampler config"""
        if request.param == 0:
            return IAFSamplerConfig(n_made_blocks=2, n_hidden_in_made=4, hidden_size=64)
        else:
            return IAFSamplerConfig(n_made_blocks=1, n_hidden_in_made=1, hidden_size=16)

    def test_fit_and_sample(self, iaf_sampler_config, model, dataset, tmp_path):
        """Check the fit method of the IAF sampler.
        We check that trying to sample before fit raises an error.
        We check that after training, the IAF sampler has the right attributes.
        We check that the sample method returns a ModelOutput with the right shape.
        """
        sampler = IAFSampler(model, iaf_sampler_config)

        dir_path = tmp_path / "dummy_folder"
        dir_path.mkdir()

        # Test that trying to sample before fit raises an error:
        with pytest.raises(ArithmeticError):
            sampler.sample(100)

        with pytest.raises(AttributeError):
            sampler.load_flows_from_folder(dir_path)

        sampler.fit(
            dataset, eval_data=dataset, training_config=BaseTrainerConfig(num_epochs=2)
        )

        assert hasattr(sampler, "flows_models")

        assert sampler.is_fitted

        if sampler.model.multiple_latent_spaces:
            for m in sampler.model.encoders:
                assert m in sampler.flows_models.keys()

        output = sampler.sample(100)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "z")
        assert output.z.shape == (100, sampler.model.latent_dim)

        if sampler.model.multiple_latent_spaces:
            assert hasattr(output, "modalities_z")

        # test save

    def test_save_and_load(self, iaf_sampler_config, model, dataset, tmp_path):
        """Test the save and load methods of the IAF sampler."""
        # Create and fit a sampler
        sampler = IAFSampler(model, iaf_sampler_config)
        dir_path = tmp_path / "dummy_folder_train"
        dir_path.mkdir()

        sampler.fit(
            dataset, eval_data=dataset, training_config=BaseTrainerConfig(num_epochs=2)
        )

        # Save the sampler and check that the files are created
        sampler.save(dir_path)
        for m in sampler.flows_models:
            assert os.path.exists(os.path.join(dir_path, m))

        # Try reloading the config and check that it is the same
        reload_config = IAFSamplerConfig.from_json_file(
            os.path.join(dir_path, "sampler_config.json")
        )
        assert reload_config == sampler.sampler_config

        # Try reloading the flows and check that they are the same
        reload_sampler = IAFSampler(model, iaf_sampler_config)
        reload_sampler.load_flows_from_folder(dir_path)

        for m, flow_model in reload_sampler.flows_models.items():
            assert all(
                [
                    torch.equal(
                        flow_model.state_dict()[key].cpu(),
                        sampler.flows_models[m].state_dict()[key].cpu(),
                    )
                    for key in flow_model.state_dict().keys()
                ]
            )
