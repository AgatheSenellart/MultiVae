import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from tests_data.classifiers import MNIST_Classifier, SVHN_Classifier
from torch import nn

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.metrics.base import Evaluator, EvaluatorConfig
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.likelihoods import (
    LikelihoodsEvaluator,
    LikelihoodsEvaluatorConfig,
)
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
def output_logger_file(tmpdir):
    os.mkdir(os.path.join(tmpdir, "logger_metrics"))
    return os.path.join(tmpdir, "logger_metrics")


class TestBaseMetric:
    @pytest.fixture(params=[10, 23])
    def batch_size(self, request):
        return request.param

    def test_evaluator_config(self, batch_size):
        config = EvaluatorConfig(batch_size=batch_size)
        assert config.batch_size == batch_size

    def test_evaluator_class(
        self, batch_size, jmvae_model, output_logger_file, dataset
    ):
        config = EvaluatorConfig(batch_size=batch_size)
        evaluator = Evaluator(
            jmvae_model, dataset, output=output_logger_file, eval_config=config
        )

        assert evaluator.n_data == len(dataset.data["mnist"])
        assert os.path.exists(os.path.join(output_logger_file, "metrics.log"))

        assert (
            next(iter(evaluator.test_loader)).data["mnist"].shape
            == (batch_size,) + dataset.data["mnist"].shape[1:]
        )


class TestCoherences:
    @pytest.fixture(params=[[True, 21], [False, 3]])
    def config_params(self, request):
        return {
            "include_recon": request.param[0],
            "nb_samples_for_joint": request.param[1],
        }

    @pytest.fixture
    def classifiers(self):
        return dict(mnist=MNIST_Classifier(), svhn=SVHN_Classifier())

    @pytest.fixture(params=[True, False])
    def sampler(self, jmvae_model, dataset, request):
        if not request.param:
            return None
        else:
            config = GaussianMixtureSamplerConfig(n_components=3)
            sampler = GaussianMixtureSampler(jmvae_model, config)
            sampler.fit(dataset)
            return sampler

    def test_coherence_config(self, config_params):
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
        )

        assert config.include_recon == config_params["include_recon"]
        assert config.nb_samples_for_joint == config_params["nb_samples_for_joint"]

    def test_cross_coherence_compute(
        self, jmvae_model, config_params, classifiers, output_logger_file, dataset
    ):
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
        )

        evaluator = CoherenceEvaluator(
            model=jmvae_model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        cross_coherences = evaluator.cross_coherences()
        assert all([0 <= cc_score[0] <= 1 for cc_score in cross_coherences])

    def test_joint_coherence_compute(
        self,
        jmvae_model,
        config_params,
        classifiers,
        output_logger_file,
        dataset,
        sampler,
    ):
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
        )

        evaluator = CoherenceEvaluator(
            model=jmvae_model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
            sampler=sampler,
        )

        joint_coherence = evaluator.joint_coherence()
        assert 0 <= joint_coherence <= 1

    def test_eval(
        self, jmvae_model, config_params, classifiers, output_logger_file, dataset
    ):
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
        )

        evaluator = CoherenceEvaluator(
            model=jmvae_model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        metrics = evaluator.eval()
        assert all(
            [
                metric in metrics.keys()
                for metric in ["mean_coherence_1", "joint_coherence_prior"]
            ]
        )


class TestLikelihoods:
    @pytest.fixture(params=[[2, 21], [16, 3]])
    def config_params(self, request):
        return {"num_samples": request.param[0], "batch_size": request.param[1]}

    @pytest.fixture
    def mopoe_model(self):
        return MoPoE(
            MoPoEConfig(
                n_modalities=2, input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32))
            )
        )

    def test_likelihood_config(self, config_params):
        config = LikelihoodsEvaluatorConfig(
            num_samples=config_params["num_samples"],
            batch_size=config_params["batch_size"],
        )

        assert config.num_samples == config_params["num_samples"]
        assert config.batch_size == config_params["batch_size"]

    def test_joint_nll(self, jmvae_model, config_params, output_logger_file, dataset):
        config = LikelihoodsEvaluatorConfig(
            num_samples=config_params["num_samples"],
            batch_size=config_params["batch_size"],
        )

        evaluator = LikelihoodsEvaluator(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        joint_nll = evaluator.joint_nll()

        assert joint_nll > 0

    def test_joint_nll_from_subset(
        self, mopoe_model, jmvae_model, config_params, output_logger_file, dataset
    ):
        config = LikelihoodsEvaluatorConfig(
            num_samples=config_params["num_samples"],
            batch_size=config_params["batch_size"],
        )

        evaluator = LikelihoodsEvaluator(
            model=mopoe_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        joint_nll_from_sub = evaluator.joint_nll_from_subset(["mnist", "svhn"])
        assert joint_nll_from_sub > 0

        evaluator = LikelihoodsEvaluator(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        joint_nll_from_sub = evaluator.joint_nll_from_subset(["mnist", "svhn"])
        assert joint_nll_from_sub is None

    def test_eval(self, jmvae_model, config_params, output_logger_file, dataset):
        config = LikelihoodsEvaluatorConfig(
            num_samples=config_params["num_samples"],
            batch_size=config_params["batch_size"],
        )

        evaluator = LikelihoodsEvaluator(
            model=jmvae_model,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        metrics = evaluator.eval()

        assert all([metric in metrics.keys() for metric in ["joint_likelihood"]])


import tempfile

from multivae.metrics import Visualization, VisualizationConfig
from multivae.models.base import ModelOutput


class Test_Visualization:
    def test(self, jmvae_model, dataset):
        tmpdir = tempfile.mkdtemp()

        module = Visualization(model=jmvae_model, output=tmpdir, test_dataset=dataset)

        output = module.eval()
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "unconditional_generation")
        assert os.path.exists(os.path.join(tmpdir, "unconditional.png"))

        output = module.conditional_samples_subset(["mnist"])
        assert isinstance(output, Image.Image)


from multivae.metrics import Clustering, ClusteringConfig


class Test_clustering:
    def test(self, jmvae_model, dataset):
        tmpdir = tempfile.mkdtemp()

        module = Clustering(
            model=jmvae_model,
            output=tmpdir,
            test_dataset=dataset,
            train_dataset=dataset,
        )

        output = module.eval()
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "cluster_accuracy")
        assert output.cluster_accuracy >= 0 and output.cluster_accuracy <= 1
