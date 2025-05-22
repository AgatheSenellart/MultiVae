import os
import shutil

import pytest
import torch
from PIL import Image
from pythae.models.base import BaseAEConfig
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_VAE_MNIST,
)
from pythae.models.nn.default_architectures import Encoder_VAE_MLP
from torch.utils.data import random_split

from multivae.data import MultimodalBaseDataset
from multivae.models import CVAE, JMVAE, TELBO, CVAEConfig, JMVAEConfig, TELBOConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import rename_logs

from .tests_data.utils import test_dataset_plotting

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def training_config(tmp_path_factory):
    """Create a training config, to test the trainer with."""
    d = tmp_path_factory.mktemp("dummy_folder")
    return BaseTrainerConfig(output_dir=str(d), num_epochs=5)


@pytest.fixture(params=[0, 20])
def model_sample_1(request):
    """Create a first model to test the trainer with."""
    model_config = JMVAEConfig(n_modalities=2, latent_dim=10, warmup=request.param)
    config = BaseAEConfig(input_dim=(1, 28, 28), latent_dim=10)
    encoders = dict(mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_VAE_MNIST(config))
    decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))
    return JMVAE(model_config=model_config, encoders=encoders, decoders=decoders)


@pytest.fixture()
def model_sample_2():
    """Create a second model to test the trainer with."""
    model_config = CVAEConfig(
        latent_dim=10,
        main_modality="mod1",
        conditioning_modalities=["mod2"],
        input_dims={"mod1": (1, 28, 28), "mod2": (1, 28, 28)},
    )
    return CVAE(model_config=model_config)


@pytest.fixture(params=["jmvae", "cvae"])
def model_sample(request, model_sample_1, model_sample_2):
    """Regroups the two models above in one fixture."""
    if request.param == "jmvae":
        return model_sample_1
    return model_sample_2


@pytest.fixture
def train_dataset_1():
    """Create a training dataset to test the trainer with."""
    return MultimodalBaseDataset(
        data=dict(
            mod1=torch.randn((2, 1, 28, 28)),
            mod2=torch.randn((2, 1, 28, 28)),
        ),
        labels=torch.tensor([0, 1]),
    )


@pytest.fixture
def train_dataset_2():
    """Create a second training dataset, which has a 'transform_for_plotting'
    function that will be used during the eval steps of the training.
    """
    return test_dataset_plotting(
        data=dict(
            mod1=torch.randn((2, 1, 28, 28)),
            mod2=torch.randn((2, 1, 28, 28)),
        ),
        labels=torch.tensor([0, 1]),
    )


class Test_Incompatible_Trainer_Model:
    """In this test, we check that initializing a BaseTrainer
    with a model that requires multistage_training raises the appropriate error.
    """

    @pytest.fixture
    def model_two_stage(self):
        """Create a model that requires two stage training."""
        model_config = TELBOConfig(
            n_modalities=2, latent_dim=10, uses_likelihood_rescaling=False
        )
        config = BaseAEConfig(input_dim=(1, 28, 28), latent_dim=10)
        encoders = dict(
            mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_VAE_MNIST(config)
        )
        decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))
        return TELBO(model_config=model_config, encoders=encoders, decoders=decoders)

    def test_setup_with_two_stage_model(
        self, model_two_stage, training_config, train_dataset_1
    ):
        """Check that an error is raised when the model is incompatible with the trainer."""
        with pytest.raises(AttributeError) as excinfo:
            BaseTrainer(
                model=model_two_stage,
                train_dataset=train_dataset_1,
                training_config=training_config,
            )
        assert "MultistageTrainer" in str(excinfo.value)


class Test_Set_Training_config:
    """In this class, we test the initialization of BaseTrainer
    with different training configurations.
    In particular, we check that the default parameters
    are used when the training configuration is None.
    """

    @pytest.fixture(
        params=[
            None,
            BaseTrainerConfig(),
            BaseTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=20,
                learning_rate=1e-5,
                optimizer_cls="AdamW",
                optimizer_params={"weight_decay": 0.01},
                scheduler_cls="ExponentialLR",
                scheduler_params={"gamma": 0.321},
            ),
        ],
        scope="class",
    )
    def training_configs(self, request, tmp_path_factory):
        """Different training configs to test."""
        if request.param is not None:
            d = tmp_path_factory.mktemp("dummy_folder")

            request.param.output_dir = str(d)
            return request.param
        else:
            return None

    def test_set_training_config(self, model_sample, train_dataset_1, training_configs):
        """Check that the training config is correctly set in the init
        of BaseTrainer.
        """
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == BaseTrainerConfig(
                output_dir="dummy_output_dir", keep_best_on_train=True
            )
            shutil.rmtree("dummy_output_dir")
        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    """In this class, we check that the optimizer is correctly instantiated
    depending on the parameters in the training configuration.
    """

    def test_wrong_optimizer_cls(self):
        """Test that wrong optimizers names raises errors."""
        with pytest.raises(AttributeError):
            BaseTrainerConfig(optimizer_cls="WrongOptim")

    def test_wrong_optimizer_params(self):
        """Check that wrong configurations for the optimizer raises an error."""
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                optimizer_cls="Adam", optimizer_params={"wrong_config": 1}
            )

    @pytest.fixture(
        params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)],
        scope="class",
    )
    def training_configs_learning_rate(self, tmp_path_factory, request):
        """Create configurations with different learning rates to test
        the optmizer instantiation
        """
        d = tmp_path_factory.mktemp("dummy_folder")
        request.param.output_dir = str(d)
        return request.param

    @pytest.fixture(
        params=[
            {"optimizer_cls": "Adagrad", "optimizer_params": {"lr_decay": 0.1}},
            {"optimizer_cls": "AdamW", "optimizer_params": {"betas": (0.1234, 0.4321)}},
            {"optimizer_cls": "SGD", "optimizer_params": None},
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):
        """Create different optimizers names and configs to test the BaseTrainer with."""
        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs_learning_rate.optimizer_params = optimizer_config[
            "optimizer_params"
        ]

        return optimizer_config

    def test_default_optimizer_building(
        self, model_sample, train_dataset_1, training_configs_learning_rate
    ):
        """Check that the default optimizer is Adam and that the learning
        rate is correctly set.
        """
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()

        assert issubclass(type(trainer.optimizer), torch.optim.Adam)
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

    def test_set_custom_optimizer(
        self,
        model_sample,
        train_dataset_1,
        training_configs_learning_rate,
        optimizer_config,
    ):
        """Check that the optimizer instantiated during BaseTrainer init
        corresponds to the parameters in the configuration.
        """
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()

        assert issubclass(
            type(trainer.optimizer),
            getattr(torch.optim, optimizer_config["optimizer_cls"]),
        )
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )
        if optimizer_config["optimizer_params"] is not None:
            assert all(
                [
                    trainer.optimizer.defaults[key]
                    == optimizer_config["optimizer_params"][key]
                    for key in optimizer_config["optimizer_params"].keys()
                ]
            )


class Test_Build_Scheduler:
    """Test that the learning rate scheduler that is instantiated in
    BaseTrainer corresponds to the parameters in the config.
    We also check the default scheduler.
    """

    def test_wrong_scheduler_cls(self):
        """Check that a wrong scheduler name raises an error."""
        with pytest.raises(AttributeError):
            BaseTrainerConfig(scheduler_cls="WrongOptim")

    def test_wrong_scheduler_params(self):
        """Check that a wrong configuration for scheduler raises an error."""
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                scheduler_cls="ReduceLROnPlateau", scheduler_params={"wrong_config": 1}
            )

    @pytest.fixture(
        params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)],
        scope="module",
    )
    def training_configs_learning_rate(self, tmp_path_factory, request):
        """Create different learning rates to test the trainer with."""
        d = tmp_path_factory.mktemp("dummy_folder")
        request.param.output_dir = str(d)
        return request.param

    @pytest.fixture(
        params=[
            {"optimizer_cls": "Adagrad", "optimizer_params": {"lr_decay": 0.1}},
            {"optimizer_cls": "AdamW", "optimizer_params": {"betas": (0.1234, 0.4321)}},
            {"optimizer_cls": "SGD", "optimizer_params": None},
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):
        """Create different optimizer to test the scheduler with."""
        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs_learning_rate.optimizer_params = optimizer_config[
            "optimizer_params"
        ]

        return optimizer_config

    @pytest.fixture(
        params=[
            {"scheduler_cls": "StepLR", "scheduler_params": {"step_size": 1}},
            {"scheduler_cls": "LinearLR", "scheduler_params": None},
            {"scheduler_cls": "ExponentialLR", "scheduler_params": {"gamma": 3.14}},
        ]
    )
    def scheduler_config(self, request, training_configs_learning_rate):
        """Create different scheduler configurations to test the trainer with."""
        scheduler_config = request.param

        # set scheduler and params to training config
        training_configs_learning_rate.scheduler_cls = scheduler_config["scheduler_cls"]
        training_configs_learning_rate.scheduler_params = scheduler_config[
            "scheduler_params"
        ]

        return request.param

    def test_default_scheduler_building(
        self, model_sample, train_dataset_1, training_configs_learning_rate
    ):
        """Check that the default scheduler is None."""
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()
        trainer.set_scheduler()

        assert trainer.scheduler is None

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset_1,
        training_configs_learning_rate,
        scheduler_config,
    ):
        """Check that the instantiated scheduler corresponds to
        the parameters in configuration.
        """
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()
        trainer.set_scheduler()

        assert issubclass(
            type(trainer.scheduler),
            getattr(torch.optim.lr_scheduler, scheduler_config["scheduler_cls"]),
        )
        if scheduler_config["scheduler_params"] is not None:
            assert all(
                [
                    trainer.scheduler.state_dict()[key]
                    == scheduler_config["scheduler_params"][key]
                    for key in scheduler_config["scheduler_params"].keys()
                ]
            )


class Test_Device_Checks:
    """Test that the device is correctly set in BaseTrainer"""

    def test_set_environ_variable(self):
        os.environ["LOCAL_RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "3"
        os.environ["MASTER_ADDR"] = "314"
        os.environ["MASTER_PORT"] = "222"

        trainer_config = BaseTrainerConfig()

        assert int(trainer_config.local_rank) == 1
        assert int(trainer_config.world_size) == 4
        assert int(trainer_config.rank) == 3
        assert trainer_config.master_addr == "314"
        assert trainer_config.master_port == "222"

        del os.environ["LOCAL_RANK"]
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]

    @pytest.fixture(
        params=[
            BaseTrainerConfig(num_epochs=3, no_cuda=True),
        ],
        scope="module",
    )
    def training_configs(self, tmp_path_factory, request):
        d = tmp_path_factory.mktemp("dummy_folder")

        request.param.output_dir = str(d)
        return request.param

    def test_setup_device_with_no_cuda(
        self, model_sample, train_dataset_1, training_configs
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            eval_dataset=train_dataset_1,
            training_config=training_configs,
        )

        device = trainer._setup_devices()
        assert device == "cpu"


class Test_set_start_keep_best_epoch:
    """For models with a start_keep_best_epoch parameter
    in their configuration, we check that this parameter is correctly set
    in the BaseTrainer.
    """

    def test(self, model_sample, train_dataset_1, training_config):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            eval_dataset=train_dataset_1,
            training_config=training_config,
        )
        if isinstance(model_sample, JMVAE):
            assert trainer.start_keep_best_epoch == model_sample.start_keep_best_epoch


class TestPredict:
    """Test the trainer's predict method wich samples generation and log them
    in a png format (when possible)
    """

    def test_predict_samples(
        self, model_sample, train_dataset_1, train_dataset_2, training_config
    ):
        """Test that samples are generated and that the output contains all modalities"""
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            eval_dataset=train_dataset_1,
            training_config=training_config,
        )

        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)

        if isinstance(model_sample, JMVAE):
            assert list(all_recons.keys()) == model_sample.modalities_name + ["all"]
        else:
            assert list(all_recons.keys()) == ["all"]

        for mod in all_recons:
            recon_mod = all_recons[mod]
            assert isinstance(recon_mod, Image.Image)
        output_without_transform = all_recons["all"]

        # Test predict on a dataset with transform_for_plotting option
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_2,
            eval_dataset=train_dataset_2,
            training_config=training_config,
        )
        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)
        output_with_transform = all_recons["all"]

        assert output_with_transform.size != output_without_transform.size

        # Test that random_split on a dataset with a transform_for_plotting doesn't raise an issue
        data1, data2 = random_split(train_dataset_2, [0.5, 0.5])
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=data1,
            eval_dataset=data2,
            training_config=training_config,
        )
        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)
        if isinstance(model_sample, JMVAE):
            assert list(all_recons.keys()) == model_sample.modalities_name + ["all"]
        else:
            assert list(all_recons.keys()) == ["all"]


class TestCreateTrainingDir:
    """We check that the training_dir is created and correspond to
    expected format.
    """

    @pytest.fixture(
        params=[
            BaseTrainerConfig(num_epochs=3, no_cuda=True),
        ]
    )
    def training_configs(self, tmp_path, request):
        d = tmp_path / "test_output_dir"

        request.param.output_dir = str(d)
        return request.param

    def test_create_dir(self, model_sample, train_dataset_1, training_configs):
        assert not os.path.exists(os.path.join(training_configs.output_dir))

        BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            eval_dataset=train_dataset_1,
            training_config=training_configs,
        )

        assert os.path.exists(os.path.join(training_configs.output_dir))


class TestLogging:
    """Check that the _get_file_logger of the trainer creates the
    right log file in the right directory.
    """

    @pytest.fixture
    def log_output_dir(self):
        return "dummy_log_output_dir"

    def test_create_dir(
        self, tmp_path, model_sample, train_dataset_1, training_config, log_output_dir
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset_1,
            eval_dataset=train_dataset_1,
            training_config=training_config,
        )

        # create dummy training signature
        trainer._training_signature = "dummy_signature"

        assert not os.path.exists(os.path.join(tmp_path, log_output_dir))
        trainer._get_file_logger(os.path.join(tmp_path, log_output_dir))

        assert os.path.exists(os.path.join(tmp_path, log_output_dir))
        assert os.path.exists(
            os.path.join(
                tmp_path, "dummy_log_output_dir", "training_logs_dummy_signature.log"
            )
        )


class TestTrainingCallbacks:
    """Test the rename_logs function."""

    def test_rename_logs(self):
        dummy_metrics = {"train_metric": 12, "eval_metric": 13}

        renamed_metrics = rename_logs(dummy_metrics)

        assert set(renamed_metrics.keys()).issubset(
            set(["train/metric", "eval/metric"])
        )


class TestSavingImages:
    """Check that images were saved in the right place"""

    def test(self, model_sample, train_dataset_1, training_config):
        training_config_predict = training_config
        training_config_predict.steps_predict = 1

        trainer = BaseTrainer(
            model_sample,
            train_dataset_1,
            eval_dataset=None,
            training_config=training_config_predict,
        )

        trainer.train()

        if isinstance(model_sample, JMVAE):
            for key in ["mod1", "mod2", "all"]:
                assert os.path.exists(
                    os.path.join(trainer.training_dir, f"recon_from_{key}.png")
                )
        else:
            assert os.path.exists(
                os.path.join(trainer.training_dir, "recon_from_all.png")
            )
