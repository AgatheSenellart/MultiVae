import os
import tempfile
from copy import deepcopy

import pytest
import torch
from pytest import fixture, mark

from multivae.data import MultimodalBaseDataset
from multivae.models.auto_model import AutoModel
from multivae.models.cvae import CVAE, CVAEConfig
from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    ConditionalDecoderMLP,
    Encoder_VAE_MLP,
    ModelOutput,
    MultipleHeadJointEncoder,
)
from multivae.models.nn.mmnist import EncoderConvMMNIST
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class TestCVAE:
    """Test class for CVAE model.
    We test the methods of the CVAE model and check training with BaseTrainer.
    """

    @fixture
    def dataset(self):
        """Create a dummy dataset for testing."""
        return MultimodalBaseDataset(
            data=dict(
                mnist=torch.randn((100, 3, 28, 28)),
                label=torch.randint(10, (100, 10)).float(),
                color=torch.randint(4, (100, 4)).float(),
            )
        )

    @fixture(params=[[10, "normal", {}, 1.0], [14, "laplace", dict(scale=0.5), 2.5]])
    def model_config(self, request):
        """Create a dummy model config for testing."""
        return CVAEConfig(
            input_dims=dict(mnist=(3, 28, 28), label=(10,), color=(4,)),
            latent_dim=request.param[0],
            conditioning_modalities=["label", "color"],
            main_modality="mnist",
            beta=request.param[3],
            decoder_dist=request.param[1],
            decoder_dist_params=request.param[2],
        )

    @fixture(params=[True, False])
    def architectures(self, model_config, request):
        """Create dummy architectures for testing."""
        if request.param:
            prior_network = MultipleHeadJointEncoder(
                dict_encoders=dict(
                    label=Encoder_VAE_MLP(
                        BaseAEConfig(
                            input_dim=(10,), latent_dim=model_config.latent_dim
                        )
                    ),
                    color=Encoder_VAE_MLP(
                        BaseAEConfig(
                            input_dim=(4,), latent_dim=model_config.latent_dim
                        ),
                    ),
                ),
                args=model_config,
                n_hidden_layers=1,
            )
            encoder = MultipleHeadJointEncoder(
                dict_encoders=dict(
                    mnist=EncoderConvMMNIST(
                        BaseAEConfig(
                            input_dim=(3, 28, 28), latent_dim=model_config.latent_dim
                        )
                    ),
                    label=Encoder_VAE_MLP(
                        BaseAEConfig(
                            input_dim=(10,), latent_dim=model_config.latent_dim
                        )
                    ),
                    color=Encoder_VAE_MLP(
                        BaseAEConfig(input_dim=(4,), latent_dim=model_config.latent_dim)
                    ),
                ),
                args=model_config,
                n_hidden_layers=1,
                hidden_dim=128,
            )
            decoder = ConditionalDecoderMLP(
                model_config.latent_dim, {"label": (10,), "color": (4,)}, (3, 28, 28)
            )
        else:
            prior_network = None
            encoder = None
            decoder = None

        return dict(encoder=encoder, decoder=decoder, prior_network=prior_network)

    def test_setup(self, model_config, architectures):
        """Test the model initialization. We check that the attributes
        are correctly set and the architectures are the right type.
        """
        model = CVAE(model_config=model_config, **architectures)

        assert model.latent_dim == model_config.latent_dim
        assert model.conditioning_modalities == model_config.conditioning_modalities
        assert model.main_modality == model_config.main_modality
        assert model.model_config == model_config

        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "prior_network")

        if architectures["encoder"] is not None:
            assert model.encoder == architectures["encoder"]
        else:
            assert isinstance(model.encoder, MultipleHeadJointEncoder)

        if architectures["decoder"] is not None:
            assert model.decoder == architectures["decoder"]
        else:
            assert isinstance(model.decoder, ConditionalDecoderMLP)

        if architectures["prior_network"] is not None:
            assert model.prior_network == architectures["prior_network"]
        else:
            assert model.prior_network is None

        return

    @fixture
    def model(self, model_config, architectures):
        """Create a dummy model for testing."""
        return CVAE(model_config=model_config, **architectures)

    def test_forward(self, model, dataset):
        """Test the forward function of the model. We check that the output is
        a ModelOutput and contains the loss.
        """
        samples = dataset[:10]
        output = model(samples)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "loss")
        assert isinstance(output.loss, torch.Tensor)

        return

    def test_encode(self, dataset, model):
        """Test the encode function of the model. We check that the output is
        a ModelOutput and contains the latent variable z.
        """
        # Generate one latent sample per datapoint
        samples = dataset[:10]
        output = model.encode(samples)

        assert isinstance(output, ModelOutput)
        assert output.z.shape == (10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert isinstance(output.cond_mod_data, dict)
        assert torch.all(output.cond_mod_data["label"] == samples.data["label"])
        assert torch.all(output.cond_mod_data["color"] == samples.data["color"])

        # Generate N=4 latent samples per datapoint
        output = model.encode(samples, N=4)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4, 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert output.cond_mod_data["label"].shape == (4, *samples.data["label"].shape)

        # Generate N=4 latent samples per datapoint and flatten the output
        output = model.encode(samples, N=4, flatten=True)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4 * 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert torch.all(
            output.cond_mod_data["label"] == torch.cat([samples.data["label"]] * 4)
        )

        return

    def test_decode(self, model, dataset):
        """Test the decode function of the model. We check that the output is
        a ModelOutput and contains the reconstruction.
        """
        samples = dataset[:10]

        embeddings = model.encode(samples)
        output = model.decode(embeddings)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "reconstruction")
        assert output.reconstruction.shape == samples.data["mnist"].shape

        return

    def test_generate_from_prior(self, model, dataset):
        """Test the generate_from_prior function of the model. We check that
        the output is a ModelOutput and contains the latent variable z.
        """
        samples = dataset[:10]

        # Define dummy conditioning variables for generation from the prior.
        cond_mod_data = dict(label=samples.data["label"], color=samples.data["color"])

        # Generate one latent sample per datapoint
        output = model.generate_from_prior(cond_mod_data=cond_mod_data)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert isinstance(output.cond_mod_data, dict)
        assert torch.all(output.cond_mod_data["label"] == samples.data["label"])

        # Generate N=4 latent samples per datapoint
        output = model.generate_from_prior(cond_mod_data=cond_mod_data, N=4)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4, 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert output.cond_mod_data["label"].shape == (4, *samples.data["label"].shape)

        # Generate N=4 latent samples per datapoint and flatten the output
        output = model.generate_from_prior(cond_mod_data, N=4, flatten=True)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4 * 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert torch.all(
            output.cond_mod_data["label"] == torch.cat([samples.data["label"]] * 4)
        )

        return

    def test_predict(self, model, dataset):
        """Test the predict function of the model.
        We check that the output contains the reconstruction and that we can predict
        using conditioning modalities only.
        """
        samples = dataset[:10]

        # Test reconstruction with the data and conditioning modalities
        output = model.predict(cond_mod="all", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "mnist")
        assert output.mnist.shape == samples.data["mnist"].shape

        output = model.predict(cond_mod=["mnist"], inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, model.main_modality)
        assert output.mnist.shape == samples.data["mnist"].shape

        output = model.predict(
            cond_mod=["mnist", "label", "color"], inputs=samples, N=10
        )
        assert isinstance(output, ModelOutput)
        assert hasattr(output, model.main_modality)
        assert output.mnist.shape == (10, 10, 3, 28, 28)

        # Test generation without the data and providing only conditioning modalities
        # This function calls generate_from_prior to generate the latent variable

        # Generate 1 sample
        output = model.predict(cond_mod=["label", "color"], inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "mnist")
        assert output.mnist.shape == samples.data["mnist"].shape
        # Generate 10 samples
        output = model.predict(cond_mod=["label", "color"], inputs=samples, N=10)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "mnist")
        assert output.mnist.shape == (10, *samples.data["mnist"].shape)
        # Generate 10 samples and flatten the output
        output = model.predict(
            cond_mod=["label", "color"], inputs=samples, N=10, flatten=True
        )
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "mnist")
        assert output.mnist.shape == (
            10 * samples.data["mnist"].shape[0],
            *samples.data["mnist"].shape[1:],
        )

        # Test that a ValueError is raised if the cond_mod is not in the conditioning modalities
        with pytest.raises(ValueError):
            model.predict(cond_mod=["color"], inputs=samples)

        with pytest.raises(ValueError):
            model.predict(cond_mod="wrong_cond_mod", inputs=samples)

        return

    @fixture(params=[[32, 64, 3, "Adagrad"], [16, 16, 4, "Adam"]])
    def trainer_config(self, request):
        """Create a dummy trainer config for testing."""
        tmp = tempfile.mkdtemp()

        return BaseTrainerConfig(
            output_dir=tmp,
            per_device_eval_batch_size=request.param[0],
            per_device_train_batch_size=request.param[1],
            num_epochs=request.param[2],
            optimizer_cls=request.param[3],
            learning_rate=1e-3,
            steps_saving=2,
        )

    @fixture
    def trainer(self, trainer_config, model, dataset):
        """Create a dummy trainer for testing."""
        return BaseTrainer(
            model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=trainer_config,
        )

    @mark.slow
    def test_train_step(self, trainer):
        """Test train step with the CVAE model."""
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

    @mark.slow
    def test_eval_step(self, trainer):
        """Test eval step with the CVAE model."""
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

    @mark.slow
    def test_main_train_loop(self, trainer):
        """Test main training loop with the CVAE model."""
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

    @mark.slow
    def test_checkpoint_saving(self, model, trainer, trainer_config):
        """Test checkpoint saving with the CVAE model.
        We check that the model and optimizer state dicts are saved correctly and can be reloaded.
        """
        dir_path = trainer_config.output_dir

        # Make a training step, save the model and reload it
        trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=1, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_1")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom architectures are in the checkpoint folder
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

        # Reload full model and check that it is the same
        model_rec = AutoModel.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert isinstance(model_rec.encoder.cpu(), type(model.encoder.cpu()))
        assert isinstance(model_rec.decoder.cpu(), type(model.decoder.cpu()))

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

    @mark.slow
    def test_checkpoint_saving_during_training(self, model, trainer, trainer_config):
        """Test the creation of a checkpoint during training loop with the CVAE model."""
        target_saving_epoch = trainer_config.steps_saving

        dir_path = trainer_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"CVAE_training_{trainer._training_signature}"
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

    @mark.slow
    def test_final_model_saving(self, model, trainer, trainer_config):
        """Test the final model saving with the CVAE model.
        We check that the model is correctly saved and can be reloaded.
        """
        dir_path = trainer_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"CVAE_training_{trainer._training_signature}"
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

        assert isinstance(model_rec.encoder.cpu(), type(model.encoder.cpu()))
        assert isinstance(model_rec.decoder.cpu(), type(model.decoder.cpu()))
