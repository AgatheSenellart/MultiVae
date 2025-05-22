import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import CMVAE, AutoModel, CMVAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_multilatents
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class TestCMVAE:
    """Test class for CMVAE model."""

    @pytest.fixture(
        params=[
            ("complete", True),
            ("complete", False),
            ("incomplete", True),
            ("incomplete", False),
        ]
    )
    def dataset(self, request):
        """Create dummy small dataset"""
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
        params=[
            (12, 2.5, 12, None, True, 6, "dreg_looser", 7),
            (
                13,
                1.2,
                3,
                {"mod1": "laplace", "mod2": "laplace", "mod3": "normal"},
                False,
                7,
                "iwae_looser",
                9,
            ),
        ]
    )
    def model_config_and_architectures(self, request):
        """Return a test model_config and custom architectures for CMVAE model"""
        model_config = CMVAEConfig(
            n_modalities=3,
            latent_dim=request.param[0],
            input_dims={"mod1": (3, 28, 28), "mod2": (3, 28, 28), "mod3": (3, 28, 28)},
            beta=request.param[1],
            modalities_specific_dim=request.param[2],
            decoders_dist=request.param[3],
            learn_modality_prior=request.param[4],
            number_of_clusters=request.param[5],
            loss=request.param[6],
            K=request.param[7],
        )

        encoders_config = BaseAEConfig(
            input_dim=(3, 28, 28),
            latent_dim=model_config.latent_dim,
            style_dim=model_config.modalities_specific_dim,
        )

        decoders_config = BaseAEConfig(
            input_dim=(3, 28, 28),
            latent_dim=model_config.latent_dim + model_config.modalities_specific_dim,
        )

        encoders = {
            m: EncoderConvMMNIST_multilatents(encoders_config)
            for m in model_config.input_dims
        }

        decoders = {
            m: DecoderConvMMNIST(decoders_config) for m in model_config.input_dims
        }

        return {
            "model_config": model_config,
            "encoders": encoders,
            "decoders": decoders,
        }

    @pytest.fixture(params=[True, False])
    def model(self, model_config_and_architectures, request):
        """Create test CMVAE model."""
        custom = request.param
        if custom:
            model = CMVAE(**model_config_and_architectures)
        else:
            model = CMVAE(model_config=model_config_and_architectures["model_config"])
        return model

    def test_setup_model(self, model, model_config_and_architectures):
        """Check that the model's attributes are setup correctly"""
        model_config = model_config_and_architectures["model_config"]

        # Check parameters setup
        assert model.n_clusters == model_config.number_of_clusters

        for mod in model_config.input_dims:
            if model_config.learn_modality_prior:
                assert model.r_logvars_priors[mod].requires_grad
            else:
                assert not model.r_logvars_priors[mod].requires_grad

    def test_forward(self, model, dataset):
        """Test the forward function of the model.
        Check that the output is a ModelOutput and that the loss is a tensor.
        """
        output = model(dataset, epoch=2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

    def test_encode(self, model, dataset, model_config_and_architectures):
        """Test the encode function of the model with different conditionning modalities.
        Check that the output is a ModelOutput with the correct attributes.
        We verify the shape of the output tensors.
        """
        model_config = model_config_and_architectures["model_config"]

        for return_mean in [True, False]:
            # conditioning on ALL modalities
            ## N=1
            outputs = model.encode(dataset[3], return_mean=return_mean)
            assert ~outputs.one_latent_space
            assert hasattr(outputs, "modalities_z")
            embeddings = outputs.z
            assert isinstance(outputs, ModelOutput)
            assert embeddings.shape == (1, model_config.latent_dim)
            # check the shape of each modalities specific variables
            for k, tensor in outputs.modalities_z.items():
                assert tensor.shape == (1, model_config.modalities_specific_dim)
            ## N>1
            outputs = model.encode(dataset[3], N=2, return_mean=return_mean)
            embeddings = outputs.z
            assert embeddings.shape == (2, 1, model_config.latent_dim)

            for k, tensor in outputs.modalities_z.items():
                assert tensor.shape == (2, 1, model_config.modalities_specific_dim)

            # conditioning on ONE modality
            ## N=1
            outputs = model.encode(dataset, cond_mod=["mod2"], return_mean=return_mean)
            embeddings = outputs.z
            assert embeddings.shape == (len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod2"].shape == (
                len(dataset),
                model_config.modalities_specific_dim,
            )
            ## N>1
            outputs = model.encode(
                dataset, cond_mod="mod3", N=10, return_mean=return_mean
            )
            embeddings = outputs.z
            assert embeddings.shape == (10, len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod3"].shape == (
                10,
                len(dataset),
                model_config.modalities_specific_dim,
            )
            # conditioning on a SUBSET of modalities
            ##N=1
            outputs = model.encode(
                dataset, cond_mod=["mod2", "mod3"], return_mean=return_mean
            )
            embeddings = outputs.z
            assert embeddings.shape == (len(dataset), model_config.latent_dim)

            assert outputs.modalities_z["mod1"].shape == (
                len(dataset),
                model_config.modalities_specific_dim,
            )

    def test_predict(self, model, dataset):
        """Test the predict function of the model.
        We check that the output is a ModelOutput and that the shape of the output tensors is correct.
        """
        ## Conditioning on ALL modalities for prediction
        Y = model.predict(dataset[3:])
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (3, 3, 28, 28)
        assert Y.mod2.shape == (3, 3, 28, 28)

        ## Conditioning on ONE modality for prediction
        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 3, 28, 28)
        assert Y.mod2.shape == (10, len(dataset), 3, 28, 28)

        ## Conditioning on a SUBSET of modalities for prediction
        Y = model.predict(dataset, cond_mod=["mod2", "mod3"], N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 3, 28, 28)
        assert Y.mod2.shape == (len(dataset) * 10, 3, 28, 28)

    def test_generate_from_prior(self, model):
        """Test the generate_from_prior function of the model.
        We check that the output is a ModelOutput and that the shape of the output tensors is correct.
        """
        # Generate one sample
        latents = model.generate_from_prior(n_samples=1)
        # Check that the output is a ModelOutput
        assert isinstance(latents, ModelOutput)
        shared = latents.z
        # Check that the shape the output tensors
        assert shared.shape == (1, model.latent_dim)
        for k, tensor in latents.modalities_z.items():
            assert tensor.shape == (1, model.model_config.modalities_specific_dim)

        # Test decode on the generated latents
        generations = model.decode(latents)

        assert isinstance(generations, ModelOutput)
        assert generations.mod1.shape == (1, 3, 28, 28)
        assert generations.mod2.shape == (1, 3, 28, 28)

        # Test with multiple generations (n = 10)
        latents = model.generate_from_prior(n_samples=10)
        assert isinstance(latents, ModelOutput)
        shared = latents.z
        assert shared.shape == (10, model.latent_dim)
        for k, tensor in latents.modalities_z.items():
            assert tensor.shape == (10, model.model_config.modalities_specific_dim)

        # Test decode on generate_from_prior
        generations = model.decode(latents)

        assert isinstance(generations, ModelOutput)
        assert generations.mod1.shape == (10, 3, 28, 28)
        assert generations.mod2.shape == (10, 3, 28, 28)

    def test_grad(self, model, dataset):
        """Check that the grad with regard to missing modalities is null and
        that the rest of the gradients are not
        """
        # Compute forward on the first 3 samples
        # where the first modality is missing in the incomplete fixture.
        output = model(dataset[:3], epoch=2)
        loss = output.loss
        loss.backward()
        # When the dataset is incomplete, the gradients of the missing modality should be null
        if isinstance(dataset, IncompleteDataset):
            for param in model.encoders["mod1"].parameters():
                assert param.grad is None or torch.all(param.grad == 0)

        # Check that when no modality are missing, the gradients are not all null
        # Compute forward on the last 3 samples without any missing modality
        output = model(dataset[-3:], epoch=2)
        loss = output.loss
        loss.backward()
        for param in model.encoders["mod1"].parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_predict_clusters(self, model, dataset):
        """Test the predict_clusters function of the model."""
        # Test with one sample
        output = model.predict_clusters(dataset[0])
        assert isinstance(output, ModelOutput)
        assert output.clusters.shape == (1,)

        # Test with a batch
        output = model.predict_clusters(dataset)
        # Check the shape and type
        assert output.clusters.shape == (len(dataset),)
        assert output.clusters.dtype == torch.int64
        # Check that the clusters values are in the range [0, n_clusters]
        assert torch.all(output.clusters <= model.n_clusters)
        assert torch.all(output.clusters >= 0)

    def test_prune_clusters(self, model, dataset):
        """Test the prune_clusters function of the model."""
        output = model.prune_clusters(dataset, batch_size=4)
        assert isinstance(output, list)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Dummy training configuration for test purposes."""
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
        """Dummy trainer for test purposes."""
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
        """Test the train_step function of the trainer with the CMVAE model."""
        start_model_state_dict = deepcopy(trainer.model.state_dict())
        start_optimizer = trainer.optimizer
        _ = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that the weights were updated after the train step
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer == start_optimizer

    @pytest.mark.slow
    def test_eval_step(self, trainer):
        """Test the eval_step function of the trainer with the CMVAE model."""
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        _ = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were not updated after the eval step
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @pytest.mark.slow
    def test_main_train_loop(self, trainer):
        """Test the main training loop with the CMVAE model."""
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated after one epoch
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @pytest.mark.slow
    def test_checkpoint_saving(self, model, trainer, training_config):
        """Test checkpoint saving with the CMVAE model.
        We check that the checkpoint is created with all the correct files and that we can reload
        the model and the optimizer from the checkpoint.
        """
        dir_path = training_config.output_dir

        # Make a training step
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        # Check that the checkpoint directory was created
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
        """Test checkpoint saving during training with the CMVAE model.
        We check that the checkpoint is created with all the correct files and that we can reload
        the model from the checkpoint.
        """
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"CMVAE_training_{trainer._training_signature}"
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
        # check that the model in the checkpoint is not the same than the final model.
        assert not all(
            [
                torch.equal(model_rec_state_dict[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    @pytest.mark.slow
    def test_final_model_saving(self, model, trainer, training_config):
        """Test the final model saving with the CMVAE model.
        We check that the final dir is created with all the correct files and that we can reload
        the correct model.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"CMVAE_training_{trainer._training_signature}"
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
        We check that the output is a tensor and that the shape is correct.
        """
        if hasattr(dataset, "masks"):
            with pytest.raises(AttributeError):
                nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
        else:
            nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
            assert nll >= 0
            assert isinstance(nll, torch.Tensor)
            assert nll.size() == torch.Size([])
