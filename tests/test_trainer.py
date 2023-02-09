import os

import pytest
import torch

from multivae.models import BaseMultiVAE, BaseMultiVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.data import MultimodalBaseDataset

PATH = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return BaseTrainerConfig(output_dir=dir_path)

@pytest.fixture
def model_sample():
    return BaseMultiVAE(BaseMultiVAEConfig())

@pytest.fixture
def train_dataset():
    return MultimodalBaseDataset()

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
        ]
    )
    def training_configs(self, request, tmpdir):
        if request.param is not None:
            tmpdir.mkdir("dummy_folder")
            dir_path = os.path.join(tmpdir, "dummy_folder")
            request.param.output_dir = dir_path
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

    @pytest.fixture(params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
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

    @pytest.fixture(params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
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
            BaseTrainerConfig(num_epochs=3, no_cuda=False),
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param
