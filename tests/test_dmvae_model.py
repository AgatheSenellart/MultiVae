import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import DMVAE, AutoModel, DMVAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_multilatents
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class TestDMVAE:
    @pytest.fixture(
        params=[
            ("complete", True),
            ("complete", False),
            ("incomplete", True),
            ("incomplete", False),
        ]
    )
    def dataset(self, request):
        """Create simple small dataset"""
        data = {
            "mod1": torch.randn((6, 3, 28, 28)),
            "mod2": torch.randn((6, 3, 28, 28)),
            "mod3": torch.randn((6, 3, 28, 28)),
        }

        labels = np.array([0, 1] * 3)
        if request.param[0] == "complete":
            if request.param[1]:
                dataset = MultimodalBaseDataset(data, labels)
            else:
                dataset = MultimodalBaseDataset(data)
        else:
            masks = dict(
                mod1=torch.Tensor([False] * 3 + [True] * 3),
                mod2=torch.Tensor([True] * 6),
                mod3=torch.Tensor([True] * 6),
            )
            if request.param[1]:
                dataset = IncompleteDataset(data=data, masks=masks, labels=labels)
            else:
                dataset = IncompleteDataset(data=data, masks=masks)

        return dataset

    @pytest.fixture(
        params=[(12, 2.5, {"mod1": 1.0, "mod2": 2.5, "mod3": 3.0}), (13, 1.2, None)]
    )
    def model_config_and_architectures(self, request):
        """Return model_config and custom architectures for DMVAE model"""
        model_config = DMVAEConfig(
            n_modalities=3,
            latent_dim=request.param[0],
            input_dims={"mod1": (3, 28, 28), "mod2": (3, 28, 28), "mod3": (3, 28, 28)},
            beta=request.param[1],
            modalities_specific_dim={"mod1": 4, "mod2": 5, "mod3": 5},
            modalities_specific_betas=request.param[2],
        )

        encoders_config = {
            m: BaseAEConfig(
                input_dim=(3, 28, 28),
                latent_dim=model_config.latent_dim,
                style_dim=model_config.modalities_specific_dim[m],
            )
            for m in model_config.modalities_specific_dim
        }

        decoders_config = {
            m: BaseAEConfig(
                input_dim=(3, 28, 28),
                latent_dim=model_config.latent_dim
                + model_config.modalities_specific_dim[m],
            )
            for m in model_config.modalities_specific_dim
        }

        encoders = {
            m: EncoderConvMMNIST_multilatents(encoders_config[m])
            for m in encoders_config
        }

        decoders = {m: DecoderConvMMNIST(decoders_config[m]) for m in decoders_config}

        return {
            "model_config": model_config,
            "encoders": encoders,
            "decoders": decoders,
        }

    @pytest.fixture(params=[True, False])
    def model(self, model_config_and_architectures, request):
        """Create DMVAE model"""
        custom = request.param
        if custom:
            model = DMVAE(**model_config_and_architectures)
        else:
            model = DMVAE(model_config=model_config_and_architectures["model_config"])
        return model

    def test_setup(self, model, model_config_and_architectures):
        """Test the setup of the model. Check attributes."""
        model_config = model_config_and_architectures["model_config"]

        assert model.beta == model_config.beta

        if model_config.modalities_specific_betas is None:
            assert model.private_betas == {"mod1": 1.0, "mod2": 1.0, "mod3": 1.0}
        else:
            assert model.private_betas == model_config.modalities_specific_betas

    def test_forward(self, model, dataset):
        """Test the forward pass of the model. Check output types and shapes."""
        output = model(dataset, epoch=2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

    def test_encode(self, model, dataset, model_config_and_architectures):
        """Test the encode function of the model. Check output types and shapes."""
        model_config = model_config_and_architectures["model_config"]

        for return_mean in [True, False]:
            # Condition on all modalities
            ## N = 1
            outputs = model.encode(dataset[3], return_mean=return_mean)
            assert ~outputs.one_latent_space
            assert hasattr(outputs, "modalities_z")
            embeddings = outputs.z
            assert isinstance(outputs, ModelOutput)
            assert embeddings.shape == (1, model_config.latent_dim)

            for k, value in model_config.modalities_specific_dim.items():
                assert outputs.modalities_z[k].shape == (1, value)
            ## N >1
            outputs = model.encode(dataset[3], N=2, return_mean=return_mean)
            embeddings = outputs.z
            assert embeddings.shape == (2, 1, model_config.latent_dim)

            for k, value in model_config.modalities_specific_dim.items():
                assert outputs.modalities_z[k].shape == (2, 1, value)

            # Condition on one modality
            ## N = 1
            outputs = model.encode(dataset, cond_mod=["mod2"], return_mean=return_mean)
            embeddings = outputs.z
            assert embeddings.shape == (len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod2"].shape == (
                len(dataset),
                model_config.modalities_specific_dim["mod2"],
            )
            ## N > 1
            outputs = model.encode(
                dataset, cond_mod="mod3", N=10, return_mean=return_mean
            )
            embeddings = outputs.z
            assert embeddings.shape == (10, len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod3"].shape == (
                10,
                len(dataset),
                model_config.modalities_specific_dim["mod3"],
            )
            # Condition on a subset of modalities
            ## N = 1
            outputs = model.encode(
                dataset, cond_mod=["mod2", "mod3"], return_mean=return_mean
            )
            embeddings = outputs.z
            assert embeddings.shape == (len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod2"].shape == (
                len(dataset),
                model_config.modalities_specific_dim["mod2"],
            )
            assert outputs.modalities_z["mod3"].shape == (
                len(dataset),
                model_config.modalities_specific_dim["mod3"],
            )

    def test_predict(self, model, dataset):
        """Test the predict function of the model. Check output types and shapes."""
        # Generate 1 sample
        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset), 3, 28, 28)
        assert Y.mod2.shape == (len(dataset), 3, 28, 28)

        # Generate 10 samples
        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 3, 28, 28)
        assert Y.mod2.shape == (10, len(dataset), 3, 28, 28)

        # Generate 10 samples and flatten
        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 3, 28, 28)
        assert Y.mod2.shape == (len(dataset) * 10, 3, 28, 28)

    def test_generate_from_prior(self, model):
        """Test the generate_from_prior function of the model. Check output types and shapes."""
        # Generate 1 latent code
        latents = model.generate_from_prior(n_samples=1)

        assert isinstance(latents, ModelOutput)
        shared = latents.z
        assert shared.shape == (model.latent_dim,)
        for mod, dim in model.style_dims.items():
            latent_mod = latents.modalities_z[mod]
            assert latent_mod.shape == (dim,)

        # Test decode on generate_from_prior
        generations = model.decode(latents)

        assert isinstance(generations, ModelOutput)
        assert generations.mod1.shape == (3, 28, 28)
        assert generations.mod2.shape == (3, 28, 28)

        # Generate 10 latent codes

        latents = model.generate_from_prior(n_samples=10)
        assert isinstance(latents, ModelOutput)
        shared = latents.z
        assert shared.shape == (10, model.latent_dim)
        for mod, dim in model.style_dims.items():
            latent_mod = latents.modalities_z[mod]
            assert latent_mod.shape == (10, dim)

        # Test decode on generate_from_prior
        generations = model.decode(latents)

        assert isinstance(generations, ModelOutput)
        assert generations.mod1.shape == (10, 3, 28, 28)
        assert generations.mod2.shape == (10, 3, 28, 28)

    def test_grad(self, model, dataset):
        """Check the gradient with regard to the missing modalities is null."""
        # Check that the gradient is null for the missing modalities
        output = model(dataset[:3], epoch=2)
        loss = output.loss
        loss.backward()
        if isinstance(dataset, IncompleteDataset):
            for param in model.encoders["mod1"].parameters():
                assert torch.all(param.grad == 0)

        # Check that the gradient when modalities are not missing
        output = model(dataset[-3:], epoch=2)
        loss = output.loss
        loss.backward()
        for param in model.encoders["mod1"].parameters():
            assert not torch.all(param.grad == 0)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create a training config for testing."""
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
        """Create a trainer for testing."""
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    @pytest.mark.slow
    def test_train_step(self, trainer):
        """Test the train step with the DMVAE model.
        We check that weights are updated.
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

    @pytest.mark.slow
    def test_eval_step(self, trainer):
        """Test eval_step with the DMVAE model.
        We check that weights are not updated.
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

    @pytest.mark.slow
    def test_main_train_loop(self, trainer):
        """Test the main training loop with the DMVAE model.
        We check that weights are updated.
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

    @pytest.mark.slow
    def test_checkpoint_saving(self, model, trainer, training_config):
        """Test checkpoint saving with the DMVAE model.
        We check that the model and optimizer state dicts are saved correctly and
        can be reloaded.
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

    @pytest.mark.slow
    def test_checkpoint_saving_during_training(self, model, trainer, training_config):
        """Test the creation of the checkpoint during the main training loop."""
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"DMVAE_training_{trainer._training_signature}"
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

    @pytest.mark.slow
    def test_final_model_saving(self, model, trainer, training_config):
        """Test final model saving with the DMVAE model.
        We check that the model and optimizer state dicts are saved correctly and
        can be reloaded.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"DMVAE_training_{trainer._training_signature}"
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
        """Test the compute_nll function of the model.
        We check that the NLL is computed correctly and is a tensor.
        """
        if hasattr(dataset, "masks"):
            with pytest.raises(AttributeError):
                nll = model.compute_nll(dataset, K=10, batch_size_K=2)
        else:
            nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])
