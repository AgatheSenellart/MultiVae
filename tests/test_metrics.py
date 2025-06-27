import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import random_split

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.metrics import Clustering, Visualization
from multivae.metrics.base import Evaluator, EvaluatorConfig
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.likelihoods import (
    LikelihoodsEvaluator,
    LikelihoodsEvaluatorConfig,
)
from multivae.models import JMVAE, JMVAEConfig, MoPoE, MoPoEConfig
from multivae.models.base import ModelOutput
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

from .tests_data.classifiers import MNIST_Classifier, SVHN_Classifier


@pytest.fixture
def jmvae_model():
    """Create a JMVAE model for testing the metrics."""
    return JMVAE(
        JMVAEConfig(
            n_modalities=2, input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32))
        )
    )


@pytest.fixture
def dataset():
    """Dummy dataset for testing."""
    return MultimodalBaseDataset(
        {"mnist": torch.randn(30, 1, 28, 28), "svhn": torch.randn(30, 3, 32, 32)},
        labels=torch.ones(30),
    )


class test_dataset_2(MultimodalBaseDataset):
    """Dataset to test the transform for plotting function"""

    def __init__(self, data, labels=None):
        super().__init__(data, labels)

    def transform_for_plotting(self, tensor, modality):
        return tensor.flatten()


@pytest.fixture
def dataset2():
    """Dummy dataset for testing the transform for plotting function."""
    return test_dataset_2(
        {"mnist": torch.randn(30, 1, 28, 28), "svhn": torch.randn(30, 3, 32, 32)},
        labels=torch.ones(30),
    )


@pytest.fixture
def output_logger_file(tmp_path):
    """Create a temporary directory for the output logger."""
    d = tmp_path / "logger_metrics"
    d.mkdir()
    return str(d)


class TestBaseMetric:
    """Test the BaseMetric class."""

    @pytest.fixture(params=[10, 23])
    def batch_size(self, request):
        """Test with different batch sizes."""
        return request.param

    def test_evaluator_config(self, batch_size):
        """Create a base evaluator configuration."""
        config = EvaluatorConfig(batch_size=batch_size)
        assert config.batch_size == batch_size

    def test_evaluator_class(
        self, batch_size, jmvae_model, output_logger_file, dataset
    ):
        """Test the init of BaseMetric. We check the attributes,
        and that an output file for logging metrics has been created.
        """
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
    """Test class for the CoherenceEvaluator class."""

    @pytest.fixture(
        params=[[True, 21, 1, False], [False, 3, 15, False], [False, 3, 34, True]]
    )
    def config_params(self, request):
        """Create a configuration for the CoherenceEvaluator."""
        return {
            "include_recon": request.param[0],
            "nb_samples_for_joint": request.param[1],
            "nb_samples_for_cross": request.param[2],
            "details_per_class": request.param[3],
        }

    @pytest.fixture
    def classifiers(self):
        """Create dummy classifiers for testing."""
        return dict(mnist=MNIST_Classifier(), svhn=SVHN_Classifier())

    @pytest.fixture(params=[True, False])
    def sampler(self, jmvae_model, dataset, request):
        """Test the metric with different samplers."""
        if not request.param:
            return None
        else:
            config = GaussianMixtureSamplerConfig(n_components=3)
            sampler = GaussianMixtureSampler(jmvae_model, config)
            sampler.fit(dataset)
            return sampler

    def test_coherence_config(self, config_params):
        """Test the init of the CoherenceEvaluatorConfig class."""
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
            give_details_per_class=config_params["details_per_class"],
            num_classes=10,
        )

        assert config.include_recon == config_params["include_recon"]
        assert config.nb_samples_for_joint == config_params["nb_samples_for_joint"]

    def test_coherence_from_subset(
        self, jmvae_model, config_params, classifiers, output_logger_file, dataset
    ):
        """The the coherence_from_subset method of the CoherenceEvaluator class.
        We check that the mean coherence is between 0 and 1,
        """
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
            give_details_per_class=config_params["details_per_class"],
            num_classes=10,
        )

        evaluator = CoherenceEvaluator(
            model=jmvae_model,
            classifiers=classifiers,
            output=output_logger_file,
            test_dataset=dataset,
            eval_config=config,
        )

        subset_dict, mean_acc = evaluator.coherence_from_subset(["mnist"])
        assert 0 <= mean_acc <= 1
        assert isinstance(subset_dict, dict)

        (
            subset_dict,
            mean_acc,
            mean_acc_per_class,
        ) = evaluator.coherence_from_subset(
            ["mnist"], return_accuracies_per_labels=True
        )

        assert isinstance(mean_acc_per_class, np.ndarray)
        assert len(mean_acc_per_class) == config.num_classes
        assert np.all(mean_acc_per_class >= 0) and np.all(mean_acc_per_class <= 1)
        assert np.allclose(np.mean(mean_acc_per_class), mean_acc)

    def test_cross_coherence_compute(
        self, jmvae_model, config_params, classifiers, output_logger_file, dataset
    ):
        """Test the cross_coherences() method.
        We check that all the coherence scores are between 0 and 1.
        """
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
            give_details_per_class=config_params["details_per_class"],
            num_classes=10,
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

        if config.give_details_per_class:
            assert all(
                [
                    (f"mean_coherence_1_class_{c}" in evaluator.metrics.keys())
                    for c in range(10)
                ]
            )

            assert all(
                [
                    (0 <= evaluator.metrics[f"mean_coherence_1_class_{c}"] <= 1)
                    for c in range(10)
                ]
            )

    def test_joint_coherence_compute(
        self,
        jmvae_model,
        config_params,
        classifiers,
        output_logger_file,
        dataset,
        sampler,
    ):
        """Test the joint_coherence method.
        We check that the joint coherence is between 0 and 1.
        """
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
            num_classes=10,
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
        """Test the eval method.
        The eval method should compute both joint and conditional coherences.
        """
        config = CoherenceEvaluatorConfig(
            include_recon=config_params["include_recon"],
            nb_samples_for_joint=config_params["nb_samples_for_joint"],
            num_classes=10,
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
    """Test class for the LikelihoodsEvaluator class."""

    @pytest.fixture(params=[[2, 21], [16, 3]])
    def config_params(self, request):
        """Create a configuration for the LikelihoodsEvaluator."""
        return {"num_samples": request.param[0], "batch_size": request.param[1]}

    @pytest.fixture
    def mopoe_model(self):
        """Create a MoPoE model for testing the metric."""
        return MoPoE(
            MoPoEConfig(
                n_modalities=2, input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32))
            )
        )

    def test_likelihood_config(self, config_params):
        """Test the init of the LikelihoodsEvaluatorConfig class."""
        config = LikelihoodsEvaluatorConfig(
            num_samples=config_params["num_samples"],
            batch_size=config_params["batch_size"],
        )

        assert config.num_samples == config_params["num_samples"]
        assert config.batch_size == config_params["batch_size"]

    def test_joint_nll(self, jmvae_model, config_params, output_logger_file, dataset):
        """Test the joint_nll method.
        We check that the returned negative log likelihood is greater than 0.
        """
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
        """Test the joint_nll_from_subset method.
        This method is not defined for all models.
        For the MoPoE model, we check that a positive value is returned, while for the JMVAE model,
         we check that None is returned.
        """
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
        """Test the eval function for the LikelihoodsEvaluator class.
        We check that the joint likelihood is computed and that the output is a ModelOutput.
        """
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


class Test_Visualization:
    """Test the Visualization class."""

    def test_saving_samples(self, jmvae_model, dataset, dataset2, tmp_path):
        """Check that the generations are sampled and saved in the right place.
        Check that the transform_for_plotting function is used.
        """
        module = Visualization(
            model=jmvae_model, output=str(tmp_path), test_dataset=dataset
        )

        output = module.eval()
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "unconditional_generation")
        assert os.path.exists(tmp_path / "unconditional.png")

        output_cond = module.conditional_samples_subset(["mnist"])
        assert isinstance(output_cond, Image.Image)

        # Test that the transform_for_plotting function is used
        module2 = Visualization(
            model=jmvae_model, output=str(tmp_path), test_dataset=dataset2
        )

        output2 = module2.eval()
        assert (
            output2.unconditional_generation.size
            != output.unconditional_generation.size
        )

        # Test that splitting the dataset with random_split doesn't cause an issue
        data1, data2 = random_split(dataset2, [0.5, 0.5])
        module3 = Visualization(
            model=jmvae_model, output=str(tmp_path), test_dataset=data1
        )

        output3 = module3.eval()
        assert isinstance(output3, ModelOutput)


class Test_clustering:
    """Test the Clustering class.
    This class can be used to test how useful the model's embeddings are for clustering the data in
    the latent space.
    """

    def test(self, jmvae_model, dataset, tmp_path):
        """Check that the clustering module returns a ModelOutput
        with a cluster_accuracy value between 0 and 1..
        """
        module = Clustering(
            model=jmvae_model,
            output=str(tmp_path),
            test_dataset=dataset,
            train_dataset=dataset,
        )

        output = module.eval()
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "cluster_accuracy")
        assert output.cluster_accuracy >= 0 and output.cluster_accuracy <= 1
