import pytest
import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.mnist import BaseAEConfig, Encoder_Conv_VAE_MNIST

from multivae.data.datasets import MultimodalBaseDataset
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
from multivae.metrics.fids.fids import AdaptShapeFID
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.svhn import Encoder_VAE_SVHN
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig


# @pytest.mark.slow
class TestFIDMetrics:
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        model_config = MVTCAEConfig(
            n_modalities=2,
            input_dims={"m0": (3, 32, 32), "m1": (1, 28, 28)},
        )
        return MVTCAE(model_config)

    @pytest.fixture
    def dataset(self):
        """Create a dummy dataset for testing"""
        return MultimodalBaseDataset(
            data={
                "m0": torch.randn((128, 3, 32, 32)),
                "m1": torch.randn((128, 1, 28, 28)),
            },
            labels=torch.ones((1024,)),
        )

    @pytest.fixture(params=[True, False])
    def sampler(self, request, model, dataset):
        """Create a GaussianMixtureSampler for testing the FIDEvaluator
        with a custom sampler.
        """
        if not request.param:
            return None
        else:
            sampler_config = GaussianMixtureSamplerConfig()
            sampler = GaussianMixtureSampler(model, sampler_config)
            sampler.fit(dataset)
            return sampler

    @pytest.fixture(params=[True, False])
    def transform(self, request):
        """Create custom transform for testing the FIDEvaluator
        with a custom transform.
        """
        return AdaptShapeFID(resize=request.param)

    def test_adapt_shape(self, transform):
        """Test the AdaptShapeFID transform for different input sizes"""
        for x in [
            torch.randn(10),
            torch.randn(10, 2),
            torch.randn(10, 20, 3),
            torch.randn(5, 6, 7, 8),
        ]:
            t_x = transform(x)
            assert len(t_x.shape) == 4
            if transform.resize:
                assert t_x.shape[-2:] == (299, 299)

        # Test that an exception is raised if the data as more than 3 dimensions (+1 batch dimension)
        x = torch.randn(3, 4, 5, 6, 7)
        with pytest.raises(AttributeError):
            t_x = transform(x)

    @pytest.fixture(params=["custom", "default"])
    def encoders_and_config(self, request, transform):
        """Create custom encoders and config for testing the FIDEvaluator."""
        if request.param == "default":
            return dict(
                eval_config=FIDEvaluatorConfig(batch_size=64),
                custom_encoders=None,
                transform=transform,
            )
        else:
            return dict(
                custom_encoders=dict(
                    m1=Encoder_Conv_VAE_MNIST(
                        BaseAEConfig(input_dim=(1, 28, 28), latent_dim=10)
                    ),
                    m0=Encoder_VAE_SVHN(
                        BaseAEConfig(input_dim=(3, 32, 32), latent_dim=12)
                    ),
                ),
                eval_config=FIDEvaluatorConfig(batch_size=64),
                transform=None,
            )

    @pytest.fixture
    def fid_model(self, model, dataset, sampler, encoders_and_config):
        """Create instance of FIDEvaluator for testing."""
        return FIDEvaluator(
            model=model,
            test_dataset=dataset,
            output=None,
            sampler=sampler,
            **encoders_and_config,
        )  # Add test with custom encoder

    def test_setup(self, fid_model, encoders_and_config):
        """Test the setup of the FIDEvaluator."""
        assert (fid_model.n_data) == 128
        assert len(fid_model.test_loader) > 0
        assert fid_model.batch_size == encoders_and_config["eval_config"].batch_size

    def test_fid_computation(self, fid_model):
        """Test the FID computation. We check the output type
        in different settings.
        """
        # Test unconditional FID
        output = fid_model.eval()
        assert isinstance(output, ModelOutput)

        # Test conditional FID
        output = fid_model.compute_fid_from_conditional_generation(
            subset=["m0"], gen_mod="m1"
        )
        assert isinstance(output, float)

        # Test all conditional FID
        output = fid_model.compute_all_conditional_fids(gen_mod="m1")
        assert isinstance(output, ModelOutput)

        assert "Conditional FD from m0 to m1" in output.__dict__.keys()
