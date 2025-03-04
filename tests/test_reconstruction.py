import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.metrics.base import Evaluator, EvaluatorConfig
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
from multivae.models import JMVAE, JMVAEConfig, MoPoE, MoPoEConfig
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig


@pytest.fixture
def jmvae_model():
    return JMVAE(
        JMVAEConfig(
            n_modalities=2, input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32))
        )
    )


@pytest.fixture
def dataset():
    return MultimodalBaseDataset(
        {"mnist": torch.randn(30, 1, 28, 28), "svhn": torch.randn(30, 3, 32, 32)},
        labels=torch.ones(30),
    )


@pytest.fixture
def output_logger_file(tmp_path):
    d = tmp_path / "logger_metrics"
    d.mkdir()
    return d


class TestReconstruction:
    @pytest.fixture(params=["SSIM", "MSE"])
    def config_params(self, request):
        return {"metric": request.param}

    def test_reconstruction_config(self, config_params):
        config = ReconstructionConfig(
            metric=config_params["metric"],
        )
        assert config.metric == config_params["metric"]

    def test_reconstruction_subset_compute(
        self, jmvae_model, config_params, output_logger_file, dataset
    ):
        config = ReconstructionConfig(metric=config_params["metric"])

        evaluator = Reconstruction(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        reconstruction_error = evaluator.reconstruction_from_subset(["mnist"])
        assert type(reconstruction_error) == torch.Tensor
        assert reconstruction_error.size() == torch.Size([])

        assert (
            f'{["mnist"]} reconstruction error ({config_params["metric"]})'
            in evaluator.metrics
        )

    def test_eval(self, jmvae_model, config_params, output_logger_file, dataset):
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
                    f'{["mnist"]} reconstruction error ({config_params["metric"]})',
                    f'{["svhn"]} reconstruction error ({config_params["metric"]})',
                    f'{list(jmvae_model.encoders.keys())} reconstruction error ({config_params["metric"]})',
                ]
            ]
        )

        evaluator.finish()
