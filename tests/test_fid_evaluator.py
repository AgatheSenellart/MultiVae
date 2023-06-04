import pytest
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets import MultimodalBaseDataset
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
from multivae.metrics.fids.fids import adapt_shape_for_fid
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.default_architectures import Encoder_VAE_MLP
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

    @pytest.fixture(params=["custom_config", "default_config"])
    def config(self, request):
        config = FIDEvaluatorConfig(
            batch_size=64, resize=False, inception_weights_path="./fids.pt"
        )
        if request.param == "custom_config":
            return config
        else:
            return FIDEvaluatorConfig(batch_size=64)

    @pytest.fixture(params=[True, False])
    def sampler(self, request, model, dataset):
        if not request.param:
            return None
        else:
            sampler_config = GaussianMixtureSamplerConfig()
            sampler = GaussianMixtureSampler(model, sampler_config)
            sampler.fit(dataset)
            return sampler

    @pytest.fixture(params=[None, lambda x: adapt_shape_for_fid(resize=False)(x)])
    def fid_model(self, model, dataset, config, sampler, request):
        return FIDEvaluator(
            model=model,
            test_dataset=dataset,
            output=None,
            eval_config=config,
            custom_encoder=None,
            transform=request.param,
            sampler=sampler,
        )  # Add test with custom encoder

    def test(self, fid_model, config):
        assert (fid_model.n_data) == 128
        assert len(fid_model.test_loader) > 0
        assert fid_model.batch_size == config.batch_size
        output = fid_model.eval()
        assert isinstance(output, ModelOutput)
