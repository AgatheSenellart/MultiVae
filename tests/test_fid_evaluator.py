import pytest
import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.mnist import BaseAEConfig, Encoder_Conv_VAE_MNIST
from torch import nn

from multivae.data.datasets import MultimodalBaseDataset
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
from multivae.metrics.fids.fids import adapt_shape_for_fid
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.default_architectures import Encoder_VAE_MLP
from multivae.models.nn.svhn import Encoder_VAE_SVHN
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig


# @pytest.mark.slow
class TestFIDMetrics:
    @pytest.fixture
    def model(self):
        model_config = MVTCAEConfig(
            n_modalities=2,
            input_dims={"m0": (3, 32, 32), "m1": (1, 28, 28)},
        )
        return MVTCAE(model_config)

    @pytest.fixture
    def dataset(self):
        return MultimodalBaseDataset(
            data={
                "m0": torch.randn((128, 3, 32, 32)),
                "m1": torch.randn((128, 1, 28, 28)),
            },
            labels=torch.ones((1024,)),
        )

    @pytest.fixture(params=[True, False])
    def sampler(self, request, model, dataset):
        if not request.param:
            return None
        else:
            sampler_config = GaussianMixtureSamplerConfig()
            sampler = GaussianMixtureSampler(model, sampler_config)
            sampler.fit(dataset)
            return sampler

    @pytest.fixture(params=["custom", "default"])
    def encoders_and_config(self, request):
        if request.param == "default":
            return dict(
                eval_config=FIDEvaluatorConfig(batch_size=64),
                custom_encoders=None,
                transform=adapt_shape_for_fid(resize=False),
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
        return FIDEvaluator(
            model=model,
            test_dataset=dataset,
            output=None,
            sampler=sampler,
            **encoders_and_config,
        )  # Add test with custom encoder

    def test(self, fid_model, encoders_and_config):
        assert (fid_model.n_data) == 128
        assert len(fid_model.test_loader) > 0
        assert fid_model.batch_size == encoders_and_config["eval_config"].batch_size
        output = fid_model.eval()
        assert isinstance(output, ModelOutput)
