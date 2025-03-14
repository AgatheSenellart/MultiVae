import os
import shutil

import pytest
import torch
import torch.optim as optim
from PIL import Image
from pydantic import ValidationError
from pythae.models.base import BaseAEConfig
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_VAE_MNIST,
)
from pythae.models.nn.default_architectures import Encoder_VAE_MLP
from torch.utils.data import random_split

from multivae.data import MultimodalBaseDataset
from multivae.models import JMVAE, TELBO, JMVAEConfig, TELBOConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import rename_logs

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def training_config(tmp_path_factory):
    d = tmp_path_factory.mktemp("dummy_folder")

    return BaseTrainerConfig(output_dir=str(d))


@pytest.fixture(params=[0, 20])
def model_sample(request):
    model_config = JMVAEConfig(n_modalities=2, latent_dim=10, warmup=request.param)
    config = BaseAEConfig(input_dim=(1, 28, 28), latent_dim=10)
    encoders = dict(mod1=Encoder_VAE_MLP(config), mod2=Encoder_Conv_VAE_MNIST(config))
    decoders = dict(mod1=Decoder_AE_MLP(config), mod2=Decoder_Conv_AE_MNIST(config))
    return JMVAE(model_config=model_config, encoders=encoders, decoders=decoders)


@pytest.fixture
def train_dataset():
    return MultimodalBaseDataset(
        data=dict(
            mod1=torch.randn((2, 1, 28, 28)),
            mod2=torch.randn((2, 1, 28, 28)),
        ),
        labels=torch.tensor([0, 1]),
    )


class test_dataset_2(MultimodalBaseDataset):
    """Dataset to test the transform for plotting function"""

    def __init__(self, data, labels=None):
        super().__init__(data, labels)

    def transform_for_plotting(self, tensor, modality):
        return tensor.flatten()


@pytest.fixture
def dataset2():
    return test_dataset_2(
        data=dict(
            mod1=torch.randn((2, 1, 28, 28)),
            mod2=torch.randn((2, 1, 28, 28)),
        ),
        labels=torch.tensor([0, 1]),
    )


class Test_Set_Trainer:
    @pytest.fixture
    def model_two_stage(self):
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
        self, model_two_stage, training_config, train_dataset
    ):
        with pytest.raises(AttributeError) as excinfo:
            trainer = BaseTrainer(
                model=model_two_stage,
                train_dataset=train_dataset,
                training_config=training_config,
            )
        assert "MultistageTrainer" in str(excinfo.value)


class Test_Set_Training_config:
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
        if request.param is not None:
            d = tmp_path_factory.mktemp("dummy_folder")

            request.param.output_dir = str(d)
            return request.param
        else:
            return None

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
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
    def test_wrong_optimizer_cls(self):
        with pytest.raises(AttributeError):
            BaseTrainerConfig(optimizer_cls="WrongOptim")

    def test_wrong_optimizer_params(self):
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                optimizer_cls="Adam", optimizer_params={"wrong_config": 1}
            )

    @pytest.fixture(
        params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)],
        scope="class",
    )
    def training_configs_learning_rate(self, tmp_path_factory, request):
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
        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs_learning_rate.optimizer_params = optimizer_config[
            "optimizer_params"
        ]

        return optimizer_config

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
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
        train_dataset,
        training_configs_learning_rate,
        optimizer_config,
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
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
    def test_wrong_scheduler_cls(self):
        with pytest.raises(AttributeError):
            BaseTrainerConfig(scheduler_cls="WrongOptim")

    def test_wrong_scheduler_params(self):
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                scheduler_cls="ReduceLROnPlateau", scheduler_params={"wrong_config": 1}
            )

    @pytest.fixture(
        params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)],
        scope="module",
    )
    def training_configs_learning_rate(self, tmp_path_factory, request):
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
        scheduler_config = request.param

        # set scheduler and params to training config
        training_configs_learning_rate.scheduler_cls = scheduler_config["scheduler_cls"]
        training_configs_learning_rate.scheduler_params = scheduler_config[
            "scheduler_params"
        ]

        return request.param

    def test_default_scheduler_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()
        trainer.set_scheduler()

        assert trainer.scheduler is None

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        scheduler_config,
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
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
        self, model_sample, train_dataset, training_configs
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        device = trainer._setup_devices()
        assert device == "cpu"


class Test_set_start_keep_best_epoch:
    def test(self, model_sample, train_dataset, training_config):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_config,
        )

        assert trainer.start_keep_best_epoch == model_sample.start_keep_best_epoch


class TestPredict:
    def test_predict_samples(
        self, model_sample, train_dataset, dataset2, training_config
    ):
        """Test that samples are generated and that the output contains all modalities"""

        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_config,
        )

        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)

        assert list(all_recons.keys()) == model_sample.modalities_name + ["all"]
        for mod in all_recons:
            recon_mod = all_recons[mod]
            assert isinstance(recon_mod, Image.Image)
        output_without_transform = all_recons["mod1"]

        # Test predict on a dataset with transform_for_plotting option
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=dataset2,
            eval_dataset=dataset2,
            training_config=training_config,
        )
        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)
        output_with_transform = all_recons["mod1"]

        assert output_with_transform.size != output_without_transform.size

        # Test that random_split on a dataset with a transform_for_plotting doesn't raise an issue
        data1, data2 = random_split(dataset2, [0.5, 0.5])
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=data1,
            eval_dataset=data2,
            training_config=training_config,
        )
        all_recons = trainer.predict(model_sample, epoch=1, n_data=3)
        assert list(all_recons.keys()) == model_sample.modalities_name + ["all"]


class TestSaving:
    @pytest.fixture(
        params=[
            BaseTrainerConfig(num_epochs=3, no_cuda=True),
        ]
    )
    def training_configs(self, tmp_path, request):
        d = tmp_path / "test_output_dir"

        request.param.output_dir = str(d)
        return request.param

    def test_create_dir(self, model_sample, train_dataset, training_configs):
        assert not os.path.exists(os.path.join(training_configs.output_dir))

        BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        assert os.path.exists(os.path.join(training_configs.output_dir))


class TestLogging:
    @pytest.fixture
    def log_output_dir(self):
        return "dummy_log_output_dir"

    def test_create_dir(
        self, tmp_path, model_sample, train_dataset, training_config, log_output_dir
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_config,
        )

        # create dummy training signature
        trainer._training_signature = "dummy_signature"

        assert not os.path.exists(os.path.join(tmp_path, "dummy_log_output_dir"))
        file_logger = trainer._get_file_logger(os.path.join(tmp_path, log_output_dir))

        assert os.path.exists(os.path.join(tmp_path, "dummy_log_output_dir"))
        assert os.path.exists(
            os.path.join(
                tmp_path, "dummy_log_output_dir", f"training_logs_dummy_signature.log"
            )
        )


class TestTrainingCallbacks:
    def test_rename_logs(self):
        dummy_metrics = {"train_metric": 12, "eval_metric": 13}

        renamed_metrics = rename_logs(dummy_metrics)

        assert set(renamed_metrics.keys()).issubset(
            set(["train/metric", "eval/metric"])
        )
