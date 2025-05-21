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
from multivae.models import JMVAE, AutoModel, JMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class TestJMVAE:
    """Main class for testing the JMVAE model."""

    @pytest.fixture()
    def dataset(self):
        """Dummy dataset."""
        data = dict(
            mod1=torch.Tensor([[1.0, 2.0], [4.0, 5.0]]),
            mod2=torch.Tensor([[67.1, 2.3, 3.0], [1.3, 2.0, 3.0]]),
            mod3=torch.Tensor([[67.1, 2.3, 3.0, 4], [1.3, 2.0, 3.0, 4]]),
        )
        labels = np.array([0, 1])
        return MultimodalBaseDataset(data, labels)

    @pytest.fixture(
        params=[
            [
                dict(
                    latent_dim=9,
                    alpha=0.1,
                    beta=0.5,
                    warmup=12,
                    input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,)),
                ),
                False,
            ],
            [dict(latent_dim=10, alpha=0.6, beta=5.0, warmup=140), True],
        ]
    )
    def config_and_architectures(self, request):
        """Create configuration for JMVAE."""
        model_config = JMVAEConfig(n_modalities=3, **request.param[0])

        if request.param[1]:
            config1 = BaseAEConfig(
                input_dim=(2,), latent_dim=request.param[0]["latent_dim"]
            )
            config2 = BaseAEConfig(
                input_dim=(3,), latent_dim=request.param[0]["latent_dim"]
            )
            config3 = BaseAEConfig(
                input_dim=(4,), latent_dim=request.param[0]["latent_dim"]
            )

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
        else:
            encoders = None
            decoders = None
        return dict(model_config=model_config, encoders=encoders, decoders=decoders)

    def test_setup(self, config_and_architectures):
        """Test initialization of the JMVAE model. Check attributes."""
        model = JMVAE(**config_and_architectures)

        # test model setup
        assert model.alpha == config_and_architectures["model_config"].alpha
        assert model.beta == config_and_architectures["model_config"].beta
        assert model.latent_dim == config_and_architectures["model_config"].latent_dim

    def test_forward(self, dataset, config_and_architectures):
        """Test forward function. Check the output."""
        model = JMVAE(**config_and_architectures)
        output = model(dataset, epoch=2, warmup=2)
        assert isinstance(output, ModelOutput)
        assert isinstance(output.loss, torch.Tensor)
        assert output.loss.size() == torch.Size([])

    def test_encode_decode(self, dataset, config_and_architectures):
        """Test the encode_function. Check that the output is
        an instance of ModelOutput and check the shape of the latent embeddings.
        Check the decode function.
        """
        model = JMVAE(**config_and_architectures)
        for return_mean in [True, False]:
            ## Encode all modalities
            # generate one latent sample conditioning on all modalities.
            outputs = model.encode(dataset, return_mean=return_mean)
            embeddings = outputs.z

            assert outputs.one_latent_space
            assert isinstance(outputs, ModelOutput)
            assert embeddings.shape == (2, model.latent_dim)

            # decode in modality mod1 and check the shape
            out_dec = model.decode(outputs, modalities="mod1")
            assert out_dec.mod1.shape == (2, 2)

            # generate 2 latent samples and flatten the output
            outputs = model.encode(dataset, N=2, flatten=True, return_mean=return_mean)
            assert outputs.z.shape == (4, model.latent_dim)

            out_dec = model.decode(outputs, modalities="mod1")
            assert out_dec.mod1.shape == (4, 2)

            ## Encode one modality
            # generate 1 sample
            embeddings = model.encode(
                dataset, cond_mod=["mod1"], return_mean=return_mean
            )
            assert embeddings.z.shape == (2, model.latent_dim)

            out_dec = model.decode(embeddings, modalities="mod1")
            assert out_dec.mod1.shape == (2, 2)

            # generate 10 samples
            embeddings = model.encode(
                dataset, cond_mod="mod2", N=10, flatten=False, return_mean=return_mean
            )
            assert embeddings.z.shape == (10, 2, model.latent_dim)

            ## Encode a subset of modalities
            embeddings = model.encode(
                dataset, cond_mod=["mod2", "mod1"], return_mean=return_mean
            )
            assert embeddings.z.shape == (2, model.latent_dim)

            out_dec = model.decode(embeddings, modalities="mod1")
            assert out_dec.mod1.shape == (2, 2)

    def test_predict(self, dataset, config_and_architectures):
        """Test the predict function of JMVAE"""
        model = JMVAE(**config_and_architectures)
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

    def test_encoder_raises_error(self, dataset, config_and_architectures):
        """Check that the encode function raises an error when
        the conditioning modalitie don't exist.
        """
        model = JMVAE(**config_and_architectures)

        with pytest.raises(AttributeError):
            model.encode(dataset, cond_mod="wrong_mod")

    @pytest.fixture()
    def incomplete_dataset(self):
        """Create an incomplete dataset for testing."""
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

    def test_error_with_incomplete_datasets(
        self, incomplete_dataset, config_and_architectures
    ):
        """Assert that the JMVAE raises error when used on
        an incomplete dataset.
        """
        model = JMVAE(**config_and_architectures)
        with pytest.raises(AttributeError):
            model(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.encode(incomplete_dataset)
        with pytest.raises(AttributeError):
            model.compute_joint_nll(incomplete_dataset, K=10, batch_size_K=2)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create a training configuration for testing."""
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
    def model(self, config_and_architectures):
        return JMVAE(**config_and_architectures)

    @pytest.fixture
    def trainer(self, model, training_config, dataset):
        """Create a trainer for testing"""
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
        """Test the train step with the JMVAE model.
        Check that the weights are updated.
        """
        start_model_state_dict = deepcopy(trainer.model.state_dict())

        _ = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    @pytest.mark.slow
    def test_eval_step(self, trainer):
        """Test the eval step with the JMVAE model. Check that the
        weights are not updated.
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
        """Test main train loop with the JMVAE model."""
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
        """Test checkpoint saving with the JMVAE model.
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
        """Test that the checkpoint is ceated during the main train loop.
        Check the checkpoint directory and saved files.
        """
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"JMVAE_training_{trainer._training_signature}"
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
        """Test final model saving with the JMVAE model.
        Check that the model is saved and can be reloaded correctly.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"JMVAE_training_{trainer._training_signature}"
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
        """Test the compute_nll function for the JMVAE model."""
        nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
        assert nll >= 0
        assert isinstance(nll, torch.Tensor)
        assert nll.size() == torch.Size([])

        cond_ll = model.compute_cond_nll(dataset, "mod1", ["mod2"])
        assert isinstance(cond_ll, dict)
        assert cond_ll["mod2"].size() == torch.Size([])
