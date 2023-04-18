import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models import JMVAE, AutoModel, JMVAEConfig
from multivae.metrics.base import EvaluatorConfig, Evaluator
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.likelihoods import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from tests_data.classifiers import SVHN_Classifier, MNIST_Classifier

@pytest.fixture
def model():
    return JMVAE(JMVAEConfig(n_modalities=2, input_dims=dict(
        mnist=(1, 28, 28),
        svhn=(3, 32, 32))
        ))

@pytest.fixture
def dataset():
    return MultimodalBaseDataset({"mnist": torch.randn(30, 1, 28, 28), "svhn": torch.randn(30, 3, 32, 32)}, labels=torch.ones(30))

@pytest.fixture
def output_logger_file(tmpdir):
    os.mkdir(os.path.join(tmpdir, 'logger_metrics'))
    return os.path.join(tmpdir, 'logger_metrics')

class TestBaseMetric:

    @pytest.fixture(params=[10, 23])
    def batch_size(self, request):
        return request.param
    
    def test_evaluator_config(self, batch_size):
        config = EvaluatorConfig(batch_size=batch_size)
        assert config.batch_size == batch_size

    def test_evaluator_class(self, batch_size, model, output_logger_file, dataset):
        config = EvaluatorConfig(batch_size=batch_size)
        evaluator = Evaluator(model, dataset, output=output_logger_file, eval_config=config)

        assert evaluator.n_data == len(dataset.data["mnist"])
        assert os.path.exists(os.path.join(output_logger_file, 'metrics.log'))

        assert next(iter(evaluator.test_loader)).data["mnist"].shape == (batch_size,) + dataset.data["mnist"].shape[1:]

class TestCoherences:

    @pytest.fixture(
            params=[
                [True, 21],
                [False, 3]
            ]
    )
    def config_params(self, request):
        return {"include_recon": request.param[0], "nb_samples_for_joint": request.param[1]}
    
    @pytest.fixture
    def classifiers(self):
        return dict(mnist=MNIST_Classifier(), svhn=SVHN_Classifier())

    def test_coherence_config(self, config_params):

        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"]
        )
        
        assert config.include_recon == config_params["include_recon"]
        assert config.nb_samples_for_joint == config_params["nb_samples_for_joint"]

    def test_cross_coherence_compute(self, model, config_params, classifiers, output_logger_file, dataset):

        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"]
        )

        evaluator = CoherenceEvaluator(
            model=model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config
        )

        cross_coherences = evaluator.cross_coherences()
        assert all([0 <= cc_score[0] <= 1 for cc_score in cross_coherences])

    def test_joint_coherence_compute(self, model, config_params, classifiers, output_logger_file, dataset):

        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"]
        )

        evaluator = CoherenceEvaluator(
            model=model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config
        )

        joint_coherence = evaluator.joint_coherence()
        assert 0 <= joint_coherence <= 1

    def test_eval(self, model, config_params, classifiers, output_logger_file, dataset):

        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"]
        )

        evaluator = CoherenceEvaluator(
            model=model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config
        )

        metrics = evaluator.eval()

        assert 0, all([metric in metrics.keys() for metric in ["means_coherences", "joint_coherence"]])

