import numpy as np
import pytest
import torch

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.mopoe.mopoe_config import MoPoEConfig
from multivae.models.mopoe.mopoe_model import MoPoE
from multivae.models.nn.default_architectures import Decoder_AE_MLP, ModelOutput
from multivae.samplers.gaussian_mixture import (
    GaussianMixtureSampler,
    GaussianMixtureSamplerConfig,
)

from .encoders import EncoderTest, EncoderTestMultilatents


class Test_GMMSampler:
    """Test the GaussianMixtureSampler class."""

    @pytest.fixture
    def dataset(self):
        """Create simple small dataset"""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
            mod4=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
        )
        labels = np.array([0, 1, 0, 0])
        dataset = MultimodalBaseDataset(data, labels)

        return dataset

    @pytest.fixture(params=["one_latent_space", "multi_latent_spaces"])
    def archi_and_config(self, beta, request):
        """Create architectures and configs for test model."""
        if request.param == "one_latent_space":
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

            model_config = MoPoEConfig(
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
            model_config = MoPoEConfig(
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
        """Beta parameter for the MoPoE model."""
        beta = request.param

        return beta

    @pytest.fixture(params=[True, False])
    def model(self, archi_and_config, request):
        """Create a MoPoE model for testing the GMM sampler."""
        custom = request.param
        if custom:
            model = MoPoE(**archi_and_config)
        else:
            model = MoPoE(archi_and_config["model_config"])
        return model

    @pytest.fixture(params=[4, 10])
    def gmm_config(self, request):
        """Create a GMM config for testing the GMM sampler."""
        return GaussianMixtureSamplerConfig(n_components=request.param)

    @pytest.fixture
    def gmm_sampler(self, gmm_config, model):
        """Create a GMM sampler."""
        sampler = GaussianMixtureSampler(model, gmm_config)
        return sampler

    def test_fit_gmm(self, gmm_sampler, dataset):
        """Check that the fit function compiles and set the right attributes"""
        gmm_sampler.fit(dataset)

        assert hasattr(gmm_sampler, "gmm")
        assert gmm_sampler.is_fitted

        if gmm_sampler.model.multiple_latent_spaces:
            assert hasattr(gmm_sampler, "mod_gmms")
            assert isinstance(gmm_sampler.mod_gmms, dict)
            assert gmm_sampler.mod_gmms.keys() == gmm_sampler.model.encoders.keys()

    def test_sample_gmm(self, gmm_sampler, dataset):
        """Check that we can sample new latent codes with the GMM sampler.
        We check the output type and the shape of the output.
        """
        gmm_sampler.fit(dataset)
        output = gmm_sampler.sample(100)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "z")
        assert output.z.shape == (100, gmm_sampler.model.latent_dim)

        if gmm_sampler.model.multiple_latent_spaces:
            assert hasattr(output, "modalities_z")
            for m, z in output.modalities_z.items():
                assert z.shape == (
                    100,
                    gmm_sampler.model.model_config.modalities_specific_dim[m],
                )
