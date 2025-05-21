import os

import numpy as np
import pytest
import torch

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import MVTCAE, MMVAEPlus, MMVAEPlusConfig, MVTCAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, ModelOutput
from multivae.samplers.maf_sampler import MAFSampler, MAFSamplerConfig

from .encoders import EncoderTest, EncoderTestMultilatents


class Test_MAFSampler:
    """Test the MAFSampler class."""

    @pytest.fixture
    def dataset(self):
        """Dummy dataset."""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
            mod4=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
        )
        labels = np.array([0, 1, 0, 0])
        dataset = MultimodalBaseDataset(data, labels)

        return dataset

    @pytest.fixture(params=[True, False])
    def one_latent_space(self, request):
        """Test the MAF sampler on one or multiple latent spaces."""
        return request.param

    @pytest.fixture
    def archi_and_config(self, beta, one_latent_space):
        """Create architectures and configs for a test model."""
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

            model_config = MVTCAEConfig(
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
            config1 = BaseAEConfig(input_dim=(2,), latent_dim=5, style_dim=3)
            config2 = BaseAEConfig(input_dim=(3,), latent_dim=5, style_dim=3)
            config3 = BaseAEConfig(input_dim=(4,), latent_dim=5, style_dim=3)

            encoders = dict(
                mod1=EncoderTestMultilatents(config1),
                mod2=EncoderTestMultilatents(config2),
                mod3=EncoderTestMultilatents(config3),
                mod4=EncoderTestMultilatents(config3),
            )
            model_config = MMVAEPlusConfig(
                n_modalities=4,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
                beta=beta,
                modalities_specific_dim=3,
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
        """Test the MAF sampler with different beta values."""
        beta = request.param

        return beta

    @pytest.fixture(params=[True, False])
    def model(self, archi_and_config, one_latent_space, request):
        """Create a test model for the MAFSampler.
        For one_latent_space, we use MVTCAE.
        For multiple_latent_spaces, we use MMVAEPlus.
        """
        custom = request.param

        if one_latent_space:
            model_class = MVTCAE
        else:
            model_class = MMVAEPlus

        if custom:
            model = model_class(**archi_and_config)
        else:
            model = model_class(archi_and_config["model_config"])
        return model

    @pytest.fixture(params=[0, 1])
    def maf_sampler_config(self, request):
        """Create a MAFSamplerConfig for testing."""
        if request.param == 0:
            return MAFSamplerConfig(n_made_blocks=2, n_hidden_in_made=4, hidden_size=64)
        else:
            return MAFSamplerConfig(n_made_blocks=1, n_hidden_in_made=1, hidden_size=16)

    def test_fit_and_sample(self, maf_sampler_config, model, dataset, tmp_path):
        """Test the MAFSampler fit function.
        We check that 1) trying to sample before fit raises an error,
        2) after fit, the sampler has the right attributes and trained modules
        3) the sample function works and returns the right output.
        """
        sampler = MAFSampler(model, maf_sampler_config)

        dir_path = tmp_path / "dummy_folder"
        dir_path.mkdir()
        # Test that trying to sample before fit raises an error:.
        with pytest.raises(ArithmeticError):
            sampler.sample(100)

        with pytest.raises(AttributeError):
            sampler.load_flows_from_folder(dir_path)

        # Fit the sampler and check post-fit attributes
        sampler.fit(dataset, eval_data=dataset)
        assert hasattr(sampler, "flows_models")
        assert sampler.is_fitted

        if sampler.model.multiple_latent_spaces:
            for m in sampler.model.encoders:
                assert m in sampler.flows_models.keys()

        # test sample, and check the shape of the output
        output = sampler.sample(100)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "z")
        assert output.z.shape == (100, sampler.model.latent_dim)

        if sampler.model.multiple_latent_spaces:
            assert hasattr(output, "modalities_z")

        # test save

    def test_save_and_load(self, maf_sampler_config, model, dataset, tmp_path):
        """Test the save and load functions of the MAFSampler.
        We check that the save function creates the right files and that
        the load function loads the models correctly.
        """
        sampler = MAFSampler(model, maf_sampler_config)
        dir_path = tmp_path / "dummy_folder"
        dir_path.mkdir()
        # Fit the sampler
        sampler.fit(dataset, eval_data=dataset)
        # Save the sampler
        sampler.save(dir_path)
        # Check that the files are created
        for m in sampler.flows_models:
            assert os.path.exists(os.path.join(dir_path, m))

        # Try reloading the config
        reload_config = MAFSamplerConfig.from_json_file(
            os.path.join(dir_path, "sampler_config.json")
        )
        assert reload_config == sampler.sampler_config

        # Try reloading the flows
        reload_sampler = MAFSampler(model, maf_sampler_config)
        reload_sampler.load_flows_from_folder(dir_path)
        # Check that the flows are loaded correctly
        for m, model in reload_sampler.flows_models.items():
            assert all(
                [
                    torch.equal(
                        model.state_dict()[key].cpu(),
                        sampler.flows_models[m].state_dict()[key].cpu(),
                    )
                    for key in model.state_dict().keys()
                ]
            )
