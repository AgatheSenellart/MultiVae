import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import TELBO, AutoModel, TELBOConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP
from multivae.trainers import BaseTrainerConfig, MultistageTrainer


class TestTELBO:
    """Class for testing the TELBO model"""

    @pytest.fixture
    def dataset(self):
        """Dummy dataset to test with"""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
        )
        labels = np.array([0, 1])
        dataset = MultimodalBaseDataset(data, labels)
        return dataset

    @pytest.fixture
    def custom_architectures(self):
        """Create architectures to test the model with"""
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config2),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config2),
        )

        return dict(encoders=encoders, decoders=decoders)

    @pytest.fixture(params=[True, False])
    def model_config(self, request):
        """Create model configurations to test the model with."""
        if request.param:
            lambda_factors = dict(mod1=4, mod2=1, mod3=3)
            gamma_factors = dict(mod1=3, mod2=3, mod3=3)

        else:
            lambda_factors = None
            gamma_factors = None
        model_config = TELBOConfig(
            n_modalities=3,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(3,)),
            lambda_factors=lambda_factors,
            gamma_factors=gamma_factors,
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        """Create different instances of the TELBO model to test."""
        custom = request.param
        if custom:
            model = TELBO(model_config, **custom_architectures)
        else:
            model = TELBO(model_config)
        return model

    def test_init(self, model, model_config):
        """Check the initialization of the model."""
        assert model.warmup == model_config.warmup

        if model_config.lambda_factors is not None:
            assert model.lambda_factors == model_config.lambda_factors
        else:
            assert model.lambda_factors == model.rescale_factors
        if model_config.gamma_factors is not None:
            assert model.gamma_factors == model_config.gamma_factors
        else:
            assert model.gamma_factors == model.rescale_factors

    def test_forward(self, model, dataset, model_config):
        """Check the forward method of the model.
        We check that the method computes and return a ModelOutput
        with the expected attributes.
        """
        # Check forward during warmup
        output = model(dataset, epoch=2)
        assert hasattr(output, "recon_loss")
        assert hasattr(output, "KLD")
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # Check forward after the warmup
        output = model(dataset, epoch=model_config.warmup + 2)
        assert not hasattr(output, "KLD")
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

    def test_encode(self, model, dataset):
        """Test the encode function of the model.
        We check that the returned embeddings are the right shape,
        depending on inputs parameters.
        """
        for return_mean in [True, False]:
            # Encode all modalities
            outputs = model.encode(dataset, return_mean=return_mean)
            assert outputs.one_latent_space
            assert isinstance(outputs, ModelOutput)
            assert outputs.z.shape == (2, 5)

            # Generate 2 latent codes per datapoint
            embeddings = model.encode(dataset, N=2, return_mean=return_mean).z
            assert embeddings.shape == (2, 2, 5)

            # Condition on one modality
            embeddings = model.encode(
                dataset, cond_mod=["mod1"], return_mean=return_mean
            ).z
            assert embeddings.shape == (2, 5)

            # Condition on one modality, generate 10 latent codes
            embeddings = model.encode(
                dataset, cond_mod="mod2", N=10, return_mean=return_mean
            ).z
            assert embeddings.shape == (10, 2, 5)

            # Condition on a subset
            embeddings = model.encode(
                dataset, cond_mod=["mod2", "mod1", "mod3"], return_mean=return_mean
            ).z
            assert embeddings.shape == (2, 5)

        # Test encode on an impossible subset
        with pytest.raises(ValueError):
            outputs = model.encode(dataset, cond_mod=["mod1", "mod2"])

    def test_predict(self, model, dataset):
        """Test the predict function of the model.
        We check that the returned reconstruction are the right shape,
        depending on inputs parameters.
        """
        Y = model.predict(dataset, cond_mod="mod1")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (2, 2)
        assert Y.mod2.shape == (2, 3)

        Y = model.predict(dataset, cond_mod="mod1", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, 2, 2)
        assert Y.mod2.shape == (10, 2, 3)

        Y = model.predict(dataset, cond_mod="mod1", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (2 * 10, 2)
        assert Y.mod2.shape == (2 * 10, 3)

    @pytest.fixture()
    def incomplete_dataset(self):
        """Create a dummy incomplete dataset, to test the model with."""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[67.1, 2.3, 3.0, 4], [1.3, 2.0, 3.0, 4]]),
        )
        masks = {
            "mod1": torch.zeros(
                2,
            ),
            "mod2": torch.zeros(
                2,
            ),
            "mod3": torch.ones(
                2,
            ),
        }
        labels = np.array([0, 1])
        return IncompleteDataset(data, labels=labels, masks=masks)

    def test_error_with_incomplete_datasets(self, incomplete_dataset, model):
        """We check that trying to use the TELBO model on a incomplte dataset
        raises the appropriate errors.
        """
        with pytest.raises(AttributeError):
            model(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.encode(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.compute_joint_nll(incomplete_dataset, K=10, batch_size_K=2)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create a training config to test the model with."""
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
        """Create a trainer to test the model with."""
        trainer = MultistageTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    def test_train_step(self, trainer):
        """Test the train step with the TELBO model.
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
        _ = trainer.prepare_train_step(10, None, None)
        _ = trainer.train_step(epoch=10)
        step_2_model_state_dict = deepcopy(trainer.model.state_dict())

        assert not all(
            [
                torch.equal(step_1_model_state_dict[key], step_2_model_state_dict[key])
                for key in step_1_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer != start_optimizer

    def test_eval_step(self, trainer):
        """Test eval_step with the TELBO model. We check that weights are not updated."""
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
        """Test the main training loop works correctly with the TELBO
        model.
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
        """We test the checkpoint saving works correctly with the
        TELBO model.
        We check that all the files are saved and that the model
        and optimizer can be correctly reloaded.
        """
        dir_path = training_config.output_dir

        # Make a training step
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        # Check that the checkpoint was created and contains all the right files
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
        """Test the create of the correct training dir and checkpoint
        during training.
        """
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"TELBO_training_{trainer._training_signature}"
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
        """Test the final model saving.
        We check that the final_directory is as expected and that the full model
        can be realoaded with AutoModel.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"TELBO_training_{trainer._training_signature}"
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
        """Test the compute_joint_nll function of the TELBO."""
        nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
        assert nll >= 0
        assert isinstance(nll, torch.Tensor)
        assert nll.size() == torch.Size([])
