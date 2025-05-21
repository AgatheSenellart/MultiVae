import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows import IAF, IAFConfig

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import JNF, AutoModel, JNFConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainerConfig, MultistageTrainer


class TestJNF:
    """Test class for the JNF model."""

    @pytest.fixture
    def dataset(self):
        """Create simple dataset"""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[67.1, 2.3, 3.0, 4], [1.3, 2.0, 3.0, 5]]),
        )
        labels = np.array([0, 1])
        dataset = MultimodalBaseDataset(data, labels)
        return dataset

    @pytest.fixture
    def custom_architectures(self):
        """Create custom architectures for testing."""
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
        )

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
        )

        flows = dict(
            mod1=IAF(IAFConfig(input_dim=(5,))),
            mod2=IAF(IAFConfig(input_dim=(5,))),
            mod3=IAF(IAFConfig(input_dim=(5,))),
        )
        return dict(
            encoders=encoders,
            decoders=decoders,
            flows=flows,
        )

    @pytest.fixture(params=[True, False])
    def use_likelihood_rescaling(self, request):
        """Test with and without rescaling."""
        return request.param

    @pytest.fixture
    def model_config(self, use_likelihood_rescaling):
        """Create model configuration for testing."""
        model_config = JNFConfig(
            n_modalities=3,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,)),
            uses_likelihood_rescaling=use_likelihood_rescaling,
        )

        return model_config

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        """Create a JNF model for test."""
        custom = request.param
        if custom:
            model = JNF(model_config, **custom_architectures)
        else:
            model = JNF(model_config)
        return model

    def test_setup(self, model, dataset, model_config):
        """Test model setup. Check the set attributes."""
        # tests on model init
        assert model.warmup == model_config.warmup

    def test_forward(self, model, dataset, model_config):
        """Check the forward function during different training stages.
        Check the output type and content.
        """
        output = model(dataset, epoch=2)
        assert hasattr(output, "metrics")

        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        assert output.metrics["ljm"] == 0

        # test model forward after warmup
        output = model(dataset, epoch=model_config.warmup + 2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        assert output.metrics["ljm"] != 0

    def test_encode_decode(self, model, dataset):
        """Test the encode function of JNF.
        Check the shape of the output depending on parameters.
        """
        for return_mean in [True, False]:
            ## Encode all modalities
            # Generate one latent sample
            outputs = model.encode(dataset, return_mean=return_mean)
            assert outputs.one_latent_space
            embeddings = outputs.z
            assert isinstance(outputs, ModelOutput)
            assert embeddings.shape == (2, 5)
            # Generate two latent samples
            embeddings = model.encode(dataset, N=2, return_mean=return_mean).z
            assert embeddings.shape == (2, 2, 5)
            ## Encode one modality
            # generate one latent sample
            embeddings = model.encode(
                dataset, cond_mod=["mod1"], return_mean=return_mean
            ).z
            assert embeddings.shape == (2, 5)
            # generate 10 latent samples
            embeddings = model.encode(
                dataset, cond_mod="mod2", N=10, return_mean=return_mean
            ).z
            assert embeddings.shape == (10, 2, 5)
            ## Encode a subset of modalities
            embeddings = model.encode(
                dataset,
                cond_mod=["mod2", "mod1"],
                mcmc_steps=2,
                return_mean=return_mean,
            ).z
            assert embeddings.shape == (2, 5)

    def test_predict(self, model, dataset):
        """Test the predict function of the JNF.
        Check the shape of the output depeding on parameters.
        """
        # Condition on one modality and reconstruct all
        Y = model.predict(dataset, cond_mod="mod1")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (2, 2)
        assert Y.mod2.shape == (2, 3)

        # Condition on one modality and reconstruct 10 times
        Y = model.predict(dataset, cond_mod="mod1", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, 2, 2)
        assert Y.mod2.shape == (10, 2, 3)

        # Condition on one modality and reconstruct 10 times but flatten
        Y = model.predict(dataset, cond_mod="mod1", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (2 * 10, 2)
        assert Y.mod2.shape == (2 * 10, 3)

    @pytest.fixture()
    def incomplete_dataset(self):
        """Dummy incomplete dataset."""
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
        """Check that the JNF model raises errors when used on incomplete data."""
        with pytest.raises(AttributeError):
            model(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.encode(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.compute_joint_nll(incomplete_dataset, K=10, batch_size_K=2)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create training config for test."""
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
        """Create trainer for test"""
        trainer = MultistageTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    def test_train_step(self, trainer):
        """Test the train step with the JNF model.
        Check that the weights are updated and that the optimizer
        is reinitialized after warmup.
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
        _ = trainer.prepare_train_step(trainer.model.warmup + 1, None, None)
        _ = trainer.train_step(epoch=trainer.model.warmup + 1)
        step_2_model_state_dict = deepcopy(trainer.model.state_dict())

        assert not all(
            [
                torch.equal(step_1_model_state_dict[key], step_2_model_state_dict[key])
                for key in step_1_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer != start_optimizer

    def test_eval_step(self, trainer):
        """Test eval step with the JNF model.
        Check that the weights are not updated.
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
        """Check training loop wih the JNF model.
        Check that the weights are updated.
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
        """Test checkpoint saving with the JNF model.
        Check that the model and optimizer are saved and can be reloaded.
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
        """Test the creation of checkpoints during the training of the JNF model"""
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"JNF_training_{trainer._training_signature}"
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
        """Test final model saving with the JNF model. Check that the model
        is saved to the right directory and can be reloaded.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"JNF_training_{trainer._training_signature}"
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
        """Test the compute_nll function of the JNF model"""
        nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
        assert nll >= 0
        assert isinstance(nll, torch.Tensor)
        assert nll.size() == torch.Size([])
