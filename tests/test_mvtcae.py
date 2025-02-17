import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base.base_model import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.auto_model.auto_model import AutoModel
from multivae.models.mvtcae import MVTCAE, MVTCAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.base_trainer_config import BaseTrainerConfig


class Test_model:
    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        # Create simple small dataset with six modalities
        data = dict(
            mod1=torch.randn((6, 2)),
            mod2=torch.randn((6, 3)),
            mod3=torch.randn((6, 4)),
            mod4=torch.randn((6, 4)),
        )
        labels = np.array([0, 1, 0, 0, 0, 0])
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data, labels)
        else:
            masks = dict(
                mod1=torch.Tensor([True] * 3 + [False] * 3),
                mod2=torch.Tensor([True] * 6),
                mod3=torch.Tensor([True] * 6),
                mod4=torch.Tensor([True] * 6),
            )
            dataset = IncompleteDataset(data=data, masks=masks, labels=labels)

        return dataset

    @pytest.fixture
    def custom_architectures(self):
        # Create an instance of mvae model
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
            mod4=Encoder_VAE_MLP(config3),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
            mod4=Decoder_AE_MLP(config3),
        )

        return dict(
            encoders=encoders,
            decoders=decoders,
        )

    @pytest.fixture(params=[1.0, 1.5, 2.0])
    def model_config(self, request):
        model_config = MVTCAEConfig(
            n_modalities=4,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            beta=request.param,
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        custom = request.param
        if custom:
            model = MVTCAE(model_config, **custom_architectures)
        else:
            model = MVTCAE(model_config)
        return model

    def test(self, model, dataset, model_config):
        assert model.beta == model_config.beta

        output = model(dataset, epoch=2)
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # Try encoding and prediction
        outputs = model.encode(dataset[0])
        assert outputs.one_latent_space
        embeddings = outputs.z
        assert isinstance(outputs, ModelOutput)
        assert embeddings.shape == (1, 5)
        embeddings = model.encode(dataset[0], N=2).z
        assert embeddings.shape == (2, 1, 5)
        embeddings = model.encode(dataset, cond_mod=["mod2"]).z
        assert embeddings.shape == (len(dataset), 5)
        embeddings = model.encode(dataset, cond_mod="mod3", N=10).z
        assert embeddings.shape == (10, len(dataset), 5)
        embeddings = model.encode(dataset, cond_mod=["mod2", "mod4"]).z
        assert embeddings.shape == (len(dataset), 5)

        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset), 2)
        assert Y.mod2.shape == (len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 2)
        assert Y.mod2.shape == (10, len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 2)
        assert Y.mod2.shape == (len(dataset) * 10, 3)


class Test_backward_with_missing_inputs:
    @pytest.fixture(params=["incomplete"])
    def dataset(self, request):
        # Create simple small dataset
        data = dict(
            mod1=torch.randn((6, 2)),
            mod2=torch.randn((6, 3)),
            mod3=torch.randn((6, 4)),
            mod4=torch.randn((6, 4)),
        )
        labels = np.array([0] * 5 + [1])
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data, labels)
        else:
            masks = dict(
                mod1=torch.Tensor([False] * 3 + [True] * 3),
                mod2=torch.Tensor([True] * 6),
                mod3=torch.Tensor([True] * 6),
                mod4=torch.Tensor([True] * 6),
            )
            dataset = IncompleteDataset(data=data, masks=masks, labels=labels)

        return dataset

    @pytest.fixture
    def custom_architectures(self):
        # Create an instance of mvae model
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
            mod4=Encoder_VAE_MLP(config3),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
            mod4=Decoder_AE_MLP(config3),
        )

        return dict(
            encoders=encoders,
            decoders=decoders,
        )

    @pytest.fixture(params=[1.0, 1.5, 2.0])
    def model_config(self, request):
        model_config = MVTCAEConfig(
            n_modalities=4,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            beta=request.param,
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        custom = request.param
        if custom:
            model = MVTCAE(model_config, **custom_architectures)
        else:
            model = MVTCAE(model_config)
        return model

    def test(self, model, dataset, model_config):
        ### Check that the grad with regard to missing modalities is null
        output = model(dataset[:3], epoch=2)
        loss = output.loss
        loss.backward()
        for param in model.encoders["mod1"].parameters():
            assert torch.all(param.grad == 0)

        output = model(dataset[-3:], epoch=2)
        loss = output.loss
        loss.backward()
        for param in model.encoders["mod1"].parameters():
            assert not torch.all(param.grad == 0)


@pytest.mark.slow
class TestTraining:
    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        # Create simple small dataset with two modalities
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
            mod4=torch.Tensor([[37, 2, 4, 1], [8, 9, 7, 0]]),
        )
        labels = np.array([0, 1])
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data, labels)
        else:
            masks = dict(
                mod1=torch.Tensor([True, False]),
                mod2=torch.Tensor([True, False]),
                mod3=torch.Tensor([True, True]),
                mod4=torch.Tensor([True, True]),
            )
            dataset = IncompleteDataset(data=data, masks=masks, labels=labels)

        return dataset

    @pytest.fixture
    def custom_architectures(self):
        # Create an instance of mmvae model
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
            mod4=Encoder_VAE_MLP(config3),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
            mod4=Decoder_AE_MLP(config3),
        )

        return dict(
            encoders=encoders,
            decoders=decoders,
        )

    @pytest.fixture(params=[0.5, 1.0, 2.0])
    def model_config(self, request):
        model_config = MVTCAEConfig(
            n_modalities=4,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            beta=request.param,
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        custom = request.param
        if custom:
            model = MVTCAE(model_config, **custom_architectures)
        else:
            model = MVTCAE(model_config)
        return model

    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return BaseTrainerConfig(
            num_epochs=3,
            steps_saving=2,
            learning_rate=1e-4,
            optimizer_cls="AdamW",
            optimizer_params={"betas": (0.91, 0.995)},
            output_dir=dir_path,
        )

    @pytest.fixture
    def trainer(self, model, training_config, dataset):
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        return trainer

    def new_trainer(self, model, training_config, dataset, checkpoint_dir):
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
            checkpoint=checkpoint_dir,
        )

        return trainer

    def test_train_step(self, trainer):
        start_model_state_dict = deepcopy(trainer.model.state_dict())
        start_optimizer = trainer.optimizer
        _ = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer == start_optimizer

    def test_eval_step(self, trainer):
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        _ = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were not updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_main_train_loop(self, trainer):
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_checkpoint_saving(self, model, trainer, training_config):
        dir_path = training_config.output_dir

        # Make a training step
        step_1_loss = trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(
            ["model.pt", "optimizer.pt", "training_config.json", "info_checkpoint.json"]
        ).issubset(set(files_list))

        # check pickled custom architectures
        for archi in model.model_config.custom_architectures:
            assert archi + ".pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert all(
            [
                torch.equal(
                    model_rec_state_dict[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        # check reload full model
        model_rec = AutoModel.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoders.cpu()) == type(model.encoders.cpu())
        assert type(model_rec.decoders.cpu()) == type(model.decoders.cpu())

        optim_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["param_groups"],
                    optimizer.state_dict()["param_groups"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["state"], optimizer.state_dict()["state"]
                )
            ]
        )

    def test_checkpoint_saving_during_training(
        self, model, trainer, training_config, dataset
    ):
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MVTCAE_training_{trainer._training_signature}"
        )

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        # try resuming
        new_trainer_ = self.new_trainer(model, training_config, dataset, checkpoint_dir)

        assert new_trainer_.best_train_loss == trainer.best_train_loss
        assert new_trainer_.trained_epochs == target_saving_epoch

        new_trainer_.train()

    def test_resume_from_checkpoint(self, model, trainer, training_config):
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MVTCAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        # check files
        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom architectures
        for archi in model.model_config.custom_architectures:
            assert archi + ".pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert not all(
            [
                torch.equal(model_rec_state_dict[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_final_model_saving(self, model, trainer, training_config):
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"MVTCAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom architectures
        for archi in model.model_config.custom_architectures:
            assert archi + ".pkl" in files_list

        # check reload full model
        model_rec = AutoModel.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoders.cpu()) == type(model.encoders.cpu())
        assert type(model_rec.decoders.cpu()) == type(model.decoders.cpu())

    def test_compute_nll(self, model, dataset):
        if not hasattr(dataset, "masks"):
            nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
            assert nll >= 0
            assert type(nll) == torch.Tensor
            assert nll.size() == torch.Size([])

            cnll = model.compute_cond_nll(
                dataset, ["mod1", "mod2"], ["mod3"], k_iwae=10
            )
            assert type(cnll) == dict
            assert "mod3" in cnll.keys()
