import pytest
import torch

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
from multivae.models import JMVAE, JMVAEConfig


@pytest.fixture
def jmvae_model():
    """Create a dummy model, to test the metric with."""
    return JMVAE(
        JMVAEConfig(
            n_modalities=2, input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32))
        )
    )


@pytest.fixture
def dataset():
    """Create a dummy dataset to test the metric with"""
    return MultimodalBaseDataset(
        {"mnist": torch.randn(30, 1, 28, 28), "svhn": torch.randn(30, 3, 32, 32)},
        labels=torch.ones(30),
    )


@pytest.fixture
def output_logger_file(tmp_path):
    """Dummy output dir, to check logging"""
    d = tmp_path / "logger_metrics"
    d.mkdir()
    return d


class TestReconstruction:
    """Test the Reconstruction metric."""

    @pytest.fixture(params=["SSIM", "MSE"])
    def config_params(self, request):
        """We test that both the Mean-Square-Error and SSIM (Structural Similarity Index are well computed)"""
        return {"metric": request.param}

    def test_reconstruction_config(self, config_params):
        """Create a configuration to test with"""
        config = ReconstructionConfig(
            metric=config_params["metric"],
        )
        assert config.metric == config_params["metric"]

    def test_reconstruction_subset_compute(
        self, jmvae_model, config_params, output_logger_file, dataset
    ):
        """We check that the reconstruction_from_subset method computes
        and returns a tensor. We check that the metric has been added to
        the evaluator's metric dict.
        """
        config = ReconstructionConfig(metric=config_params["metric"])

        evaluator = Reconstruction(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        reconstruction_error = evaluator.reconstruction_from_subset(["mnist"])
        assert isinstance(reconstruction_error, torch.Tensor)
        assert reconstruction_error.size() == torch.Size([])

        assert (
            f"{['mnist']} reconstruction error ({config_params['metric']})"
            in evaluator.metrics
        )

    def test_eval(self, jmvae_model, config_params, output_logger_file, dataset):
        """We check that the eval method computes and returns a dictionary
        with all the expected keys.
        """
        config = ReconstructionConfig(metric=config_params["metric"])

        evaluator = Reconstruction(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        metrics = evaluator.eval()
        assert all(
            [
                key in metrics.keys()
                for key in [
                    f"{['mnist']} reconstruction error ({config_params['metric']})",
                    f"{['svhn']} reconstruction error ({config_params['metric']})",
                    f"{list(jmvae_model.encoders.keys())} reconstruction error ({config_params['metric']})",
                ]
            ]
        )

        evaluator.finish()
