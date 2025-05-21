import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import MMVAE, AutoModel, MMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class TestMMVAE:
    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        """Dummy dataset."""
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
        """Create custom architectures for the model."""
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)
        config4 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
            mod4=Encoder_VAE_MLP(config4),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
            mod4=Decoder_AE_MLP(config4),
        )

        return dict(
            encoders=encoders,
            decoders=decoders,
        )

    @pytest.fixture(params=[(True, "iwae_looser"), (False, "dreg_looser")])
    def model_config(self, request):
        """Create model config for the test."""
        model_config = dict(
            n_modalities=4,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            use_likelihood_rescaling=request.param,
            decoders_dist=dict(
                mod1="laplace", mod2="laplace", mod3="laplace", mod4="laplace"
            ),
            decoder_dist_params=dict(mod1={"scale": 0.75}, mod2={"scale": 0.75}),
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        """Create MMVAE model for the test."""
        custom = request.param
        if custom:
            model = MMVAE(MMVAEConfig(**model_config), **custom_architectures)
        else:
            model = MMVAE(MMVAEConfig(**model_config))
        return model

    def test_init(self, model, dataset, model_config):
        """Test the model initialization."""
        model_config = MMVAEConfig(**model_config)
        assert model_config.decoders_dist == dict(
            mod1="laplace", mod2="laplace", mod3="laplace", mod4="laplace"
        )

        assert model_config.decoder_dist_params == dict(
            mod1={"scale": 0.75}, mod2={"scale": 0.75}
        )

        # Try forward

    def test_forward(self, model, dataset):
        """Test the model forward pass."""
        output = model(dataset, epoch=2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

    def test_encode(self, model, dataset):
        """Test the model encode method"""
        for return_mean in [True, False]:
            # generate one latent code and check the shape
            outputs = model.encode(
                dataset, ignore_incomplete=True, return_mean=return_mean
            )
            assert outputs.one_latent_space
            embeddings = outputs.z
            assert isinstance(outputs, ModelOutput)
            assert embeddings.shape == (len(dataset), 5)
            # generate 2 latent code and check the shape
            embeddings = model.encode(
                dataset, N=2, ignore_incomplete=True, return_mean=return_mean
            ).z
            assert embeddings.shape == (2, len(dataset), 5)
            # generate 1 latent code for one modality and check the shape
            embeddings = model.encode(
                dataset,
                cond_mod=["mod1"],
                ignore_incomplete=True,
                return_mean=return_mean,
            ).z
            assert embeddings.shape == (len(dataset), 5)
            # generate 10 latent code for one modality and check the shape
            embeddings = model.encode(
                dataset, cond_mod="mod2", N=10, return_mean=return_mean
            ).z
            assert embeddings.shape == (10, len(dataset), 5)

            # generate 1 latent code for two modalities and check the shape
            embeddings = model.encode(
                dataset,
                cond_mod=["mod2", "mod1"],
                ignore_incomplete=True,
                return_mean=return_mean,
            ).z
            assert embeddings.shape == (len(dataset), 5)

    def test_predict(self, model, dataset):
        """Test the predict method of the model."""
        Y = model.predict(dataset, cond_mod="mod1", ignore_incomplete=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset), 2)
        assert Y.mod2.shape == (len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod1", N=10, ignore_incomplete=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 2)
        assert Y.mod2.shape == (10, len(dataset), 3)

        Y = model.predict(
            dataset, cond_mod="mod1", N=10, flatten=True, ignore_incomplete=True
        )
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 2)
        assert Y.mod2.shape == (len(dataset) * 10, 3)

    def test_backward_with_missing_mods(self, model, dataset):
        """Check that the grad with regard to missing modalities is null"""
        if isinstance(dataset, IncompleteDataset):
            output = model(dataset[:3], epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert param.grad is None or torch.all(param.grad == 0)

            output = model(dataset, epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert not torch.all(param.grad == 0)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create a training config for the test."""
        dir_path = tmp_path_factory.mktemp("dummy_folder")

        yield BaseTrainerConfig(
            num_epochs=3,
            steps_saving=2,
            learning_rate=1e-4,
            optimizer_cls="AdamW",
            optimizer_params={"betas": (0.91, 0.995)},
            output_dir=str(dir_path),
            no_cuda=True,
        )
        shutil.rmtree(dir_path)

    @pytest.fixture
    def trainer(self, model, training_config, dataset):
        """Create a trainer for the test."""
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    def test_train_step(self, trainer):
        """Test the train step with the MMVAE model.
        weights should be updated
        """
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
        """Test eval step with the MMVAE model.
        weights should not be updated
        """
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
        """Test the main training loop with the MMVAE model.
        weights should be updated
        """
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
        """Test checkpoint saving with the MMVAE model.
        We check that the model and optimizer are correctly saved
        and can be reloaded.
        """
        dir_path = training_config.output_dir

        # Make a training step
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

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

        assert isinstance(model_rec.encoders.cpu(), type(model.encoders.cpu()))
        assert isinstance(model_rec.decoders.cpu(), type(model.decoders.cpu()))

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

    def test_checkpoint_saving_during_training(self, model, trainer, training_config):
        """Test the creation of a checkpoint during training."""
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MMVAE_training_{trainer._training_signature}"
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
        """Test final saving for the mmvae model."""
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"MMVAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, "final_model")
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

        assert isinstance(model_rec.encoders.cpu(), type(model.encoders.cpu()))
        assert isinstance(model_rec.decoders.cpu(), type(model.decoders.cpu()))

    def test_compute_nll(self, model, dataset):
        """Test the compute_joint_nll method of the MMVAE model."""
        if not hasattr(dataset, "masks"):
            nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])

            nll = model.compute_joint_nll_paper(dataset, K=10, batch_size_K=2)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])
