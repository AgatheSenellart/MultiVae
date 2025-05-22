import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models import AutoModel, Nexus, NexusConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainer, BaseTrainerConfig

from .encoders import DecoderTest, EncoderTest


class TestNexus:
    """Class for testing the Nexus model"""

    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        """Create a dummy dataset"""
        data = dict(
            mod1=torch.Tensor(np.random.random((4, 1, 12, 12))),
            mod2=torch.Tensor(np.random.random((4, 3, 7, 7))),
            mod3=torch.Tensor(np.random.random((4, 3))),
            mod4=torch.Tensor(np.random.random((4, 5))),
        )
        labels = np.array([0, 1, 0, 1])
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data, labels)
        else:
            masks = dict(
                mod1=torch.Tensor([True, True, False, False]),
                mod2=torch.Tensor([True, True, True, True]),
                mod3=torch.Tensor([True, True, True, True]),
                mod4=torch.Tensor([True, True, True, True]),
            )
            dataset = IncompleteDataset(data=data, masks=masks, labels=labels)

        return dataset

    @pytest.fixture(
        params=[
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=14,
                latent_dim=3,
                uses_likelihood_rescaling=True,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=14,
                latent_dim=3,
                uses_likelihood_rescaling=True,
                rescale_factors=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=None,
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=12,
                latent_dim=3,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=None,
                msg_dim=13,
                latent_dim=3,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
                dropout_rate=0.5,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
                dropout_rate=1,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
                top_beta=3,
            ),
            dict(
                n_modalities=4,
                input_dims=dict(
                    mod1=(1, 12, 12),
                    mod2=(3, 7, 7),
                    mod3=(3,),
                    mod4=(5,),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
                top_beta=3,
                adapt_top_decoder_variance=["mod1"],
            ),
        ]
    )
    def custom_config_archi(self, request):
        """ "Create a model configuration"""
        encoders = dict()
        decoders = dict()
        top_encoders = dict()
        top_decoders = dict()
        for m in request.param["modalities_specific_dim"]:
            encoders[m] = EncoderTest(
                BaseAEConfig(
                    input_dim=request.param["input_dims"][m],
                    latent_dim=request.param["modalities_specific_dim"][m],
                )
            )

            decoders[m] = DecoderTest(
                BaseAEConfig(
                    input_dim=request.param["input_dims"][m],
                    latent_dim=request.param["modalities_specific_dim"][m],
                )
            )

            top_encoders[m] = EncoderTest(
                BaseAEConfig(
                    input_dim=(request.param["modalities_specific_dim"][m],),
                    latent_dim=request.param["msg_dim"],
                )
            )

            top_decoders[m] = DecoderTest(
                BaseAEConfig(
                    input_dim=(request.param["modalities_specific_dim"][m],),
                    latent_dim=request.param["latent_dim"],
                )
            )

        joint_encoder = EncoderTest(
            BaseAEConfig(
                input_dim=(request.param["msg_dim"],),
                latent_dim=request.param["latent_dim"],
            )
        )

        model_config = NexusConfig(**request.param)

        for key in request.param:
            assert model_config.to_dict()[key] == request.param[key]

        return dict(
            encoders=encoders,
            decoders=decoders,
            top_encoders=top_encoders,
            top_decoders=top_decoders,
            joint_encoder=joint_encoder,
            model_config=model_config,
        )

    def test_model_setup_with_custom_architectures(self, custom_config_archi):
        """Test the model initialization with custom architectures."""
        model = Nexus(**custom_config_archi)

        if custom_config_archi["model_config"].gammas is not None:
            assert model.gammas == custom_config_archi["model_config"].gammas

        else:
            assert model.gammas == {m: 1.0 for m in model.encoders}

        if custom_config_archi["model_config"].bottom_betas is not None:
            assert (
                model.bottom_betas == custom_config_archi["model_config"].bottom_betas
            )

        else:
            assert model.bottom_betas == {m: 1.0 for m in model.encoders}

        if custom_config_archi["model_config"].uses_likelihood_rescaling:
            assert model.use_likelihood_rescaling
            assert model.rescale_factors is not None
            if custom_config_archi["model_config"].rescale_factors is not None:
                assert (
                    model.rescale_factors
                    == custom_config_archi["model_config"].rescale_factors
                )

        assert all(
            [
                model.encoders[m] == custom_config_archi["encoders"][m]
                for m in model.encoders
            ]
        )

        assert all(
            [
                model.decoders[m] == custom_config_archi["decoders"][m]
                for m in model.decoders
            ]
        )

        assert all(
            [
                model.top_encoders[m] == custom_config_archi["top_encoders"][m]
                for m in model.encoders
            ]
        )

        assert all(
            [
                model.top_decoders[m] == custom_config_archi["top_decoders"][m]
                for m in model.encoders
            ]
        )

        assert model.joint_encoder == custom_config_archi["joint_encoder"]

    def test_model_setup_without_custom_architectures(self, custom_config_archi):
        """Test the model initialization without custom architectures.
        In that case, default architectures should be used.
        """
        model = Nexus(model_config=custom_config_archi["model_config"])

        if custom_config_archi["model_config"].gammas is not None:
            assert model.gammas == custom_config_archi["model_config"].gammas

        else:
            assert model.gammas == {m: 1.0 for m in model.encoders}

        if custom_config_archi["model_config"].bottom_betas is not None:
            assert (
                model.bottom_betas == custom_config_archi["model_config"].bottom_betas
            )

        else:
            assert model.bottom_betas == {m: 1.0 for m in model.encoders}

        if custom_config_archi["model_config"].uses_likelihood_rescaling:
            assert model.use_likelihood_rescaling
            assert model.rescale_factors is not None
            if custom_config_archi["model_config"].rescale_factors is not None:
                assert (
                    model.rescale_factors
                    == custom_config_archi["model_config"].rescale_factors
                )

        assert all(
            [isinstance(model.encoders[m], Encoder_VAE_MLP) for m in model.encoders]
        )

        assert all(
            [isinstance(model.decoders[m], Decoder_AE_MLP) for m in model.encoders]
        )

        assert all(
            [isinstance(model.top_encoders[m], Encoder_VAE_MLP) for m in model.encoders]
        )

        assert all(
            [isinstance(model.top_decoders[m], Decoder_AE_MLP) for m in model.encoders]
        )

        assert isinstance(model.joint_encoder, Encoder_VAE_MLP)

    def test_setup_with_wrong_attributes(self, custom_config_archi):
        """Check that init raises errors when the attributes
        are not correct
        """
        dict_config = custom_config_archi["model_config"].to_dict()
        dict_config.pop("name")

        # wrong aggregator
        wrong_config = NexusConfig(**dict_config)
        wrong_config.aggregator = "concat"
        with pytest.raises(AttributeError):
            Nexus(model_config=wrong_config)

        # No input_dims and no encoders
        wrong_config = NexusConfig(**dict_config)
        wrong_config.input_dims = None
        with pytest.raises(AttributeError):
            Nexus(
                model_config=wrong_config,
                decoders=custom_config_archi["decoders"],
                top_decoders=custom_config_archi["top_decoders"],
                top_encoders=custom_config_archi["top_encoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )

        # No input dims and no decoders
        with pytest.raises(AttributeError):
            Nexus(
                model_config=wrong_config,
                encoders=custom_config_archi["encoders"],
                top_decoders=custom_config_archi["top_decoders"],
                top_encoders=custom_config_archi["top_encoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )

        # No modalities_dims and no top_decoders
        wrong_config = NexusConfig(**dict_config)
        wrong_config.modalities_specific_dim = None
        with pytest.raises(AttributeError):
            Nexus(
                model_config=wrong_config,
                decoders=custom_config_archi["decoders"],
                encoders=custom_config_archi["encoders"],
                top_encoders=custom_config_archi["top_encoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )
        # No modalities_dims and no top_decoders
        with pytest.raises(AttributeError):
            Nexus(
                model_config=wrong_config,
                decoders=custom_config_archi["decoders"],
                encoders=custom_config_archi["encoders"],
                top_decoders=custom_config_archi["top_decoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )
        # Top encoder is not BaseEncoder
        wrong_top_encoders = custom_config_archi["top_encoders"]
        wrong_top_encoders["mod1"] = nn.Linear(3, 4)
        with pytest.raises(AttributeError):
            Nexus(
                model_config=NexusConfig(**dict_config),
                decoders=custom_config_archi["decoders"],
                encoders=custom_config_archi["encoders"],
                top_encoders=wrong_top_encoders,
                top_decoders=custom_config_archi["top_decoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )
        # Top decoder is not BaseDecoder
        wrong_top_decoders = custom_config_archi["top_decoders"]
        wrong_top_decoders["mod1"] = nn.Linear(3, 4)
        with pytest.raises(AttributeError):
            Nexus(
                model_config=NexusConfig(**dict_config),
                decoders=custom_config_archi["decoders"],
                encoders=custom_config_archi["encoders"],
                top_decoders=wrong_top_decoders,
                top_encoders=custom_config_archi["top_encoders"],
                joint_encoder=custom_config_archi["joint_encoder"],
            )
        # Joint Encoder is not BaseEncoder
        wrong_joint_encoder = nn.Linear(3, 4)
        with pytest.raises(AttributeError):
            Nexus(
                model_config=NexusConfig(**dict_config),
                decoders=custom_config_archi["decoders"],
                encoders=custom_config_archi["encoders"],
                top_decoders=wrong_top_decoders,
                top_encoders=custom_config_archi["top_encoders"],
                joint_encoder=wrong_joint_encoder,
            )
        # wrong_gammas
        wrong_model_config = NexusConfig(**dict_config)
        if wrong_model_config.gammas is not None:
            wrong_model_config.gammas.pop("mod1")
            with pytest.raises(AttributeError):
                Nexus(model_config=wrong_model_config)
        # wrong_betas
        wrong_model_config = NexusConfig(**dict_config)
        if wrong_model_config.bottom_betas is not None:
            wrong_model_config.bottom_betas.pop("mod1")
            with pytest.raises(AttributeError):
                Nexus(model_config=wrong_model_config)

    @pytest.fixture(params=["custom_architectures", "default_architectures"])
    def model(self, custom_config_archi, request):
        """Create the model"""
        if request.param == "custom_architectures":
            return Nexus(**custom_config_archi)
        else:
            return Nexus(model_config=custom_config_archi["model_config"])

    def test_forward(self, model, dataset):
        """Test the forward method for Nexus.
        We check that the output is a ModelOutput with the loss tensor.
        """
        output = model(dataset, epoch=2)
        loss = output.loss
        assert isinstance(loss, torch.Tensor)
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

    def test_encode(self, model, dataset):
        """Test the encode function for Nexus.
        We check the shape of the latent variables, depending on parameters.
        """
        for return_mean in [True, False]:
            # Encode one sample
            outputs = model.encode(dataset[0], return_mean=return_mean)
            assert outputs.one_latent_space
            assert isinstance(outputs, ModelOutput)
            assert outputs.z.shape == (1, model.latent_dim)

            # Encode one sample, generate two latent codes
            embeddings = model.encode(dataset[0], N=2, return_mean=return_mean).z
            assert embeddings.shape == (2, 1, model.latent_dim)

            # Encode the dataset conditioning on 1 modality
            embeddings = model.encode(
                dataset, cond_mod=["mod2"], return_mean=return_mean
            ).z
            assert embeddings.shape == (4, model.latent_dim)

            # Encode the dataset conditioning on one modality/ generate 10 latents
            embeddings = model.encode(
                dataset, cond_mod="mod3", N=10, return_mean=return_mean
            ).z
            assert embeddings.shape == (10, 4, model.latent_dim)

            # Encode the dataset conditioning on a subset of modalities
            embeddings = model.encode(
                dataset, cond_mod=["mod2", "mod4"], return_mean=return_mean
            ).z
            assert embeddings.shape == (4, model.latent_dim)

    def test_decode(self, model, dataset):
        """Test the decode method of Nexus.
        We check the reconstruction shapes
        """
        # Test decode
        Y = model.decode(model.encode(dataset, cond_mod="mod3", N=10))
        assert Y.mod1.shape == (10, 4, 1, 12, 12)

    def test_predict(self, model, dataset):
        """Test the predict method of the model.
        We check the reconstruction shapes depending on the parameters.
        """
        # test predict conditioning on one modality
        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (4, 1, 12, 12)
        assert Y.mod2.shape == (4, 3, 7, 7)

        # test predic conditioning on one modality, generating 10 recontructions
        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, 4, 1, 12, 12)
        assert Y.mod2.shape == (10, 4, 3, 7, 7)

        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (4 * 10, 1, 12, 12)
        assert Y.mod2.shape == (4 * 10, 3, 7, 7)

    def test_grad_with_missing_inputs(self, model, dataset):
        """Check that the grad with regard to missing modality is 0"""
        if hasattr(dataset, "masks"):
            output = model(dataset[2:], epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert torch.all(param.grad == 0)

            output = model(dataset[:2], epoch=2)
            loss = output.loss
            loss.backward()
            for param in model.encoders["mod1"].parameters():
                assert not torch.all(param.grad == 0)

    @pytest.fixture
    def training_config(self, tmp_path_factory):
        """Create training configuration for testing the Nexus Model"""
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
        """Create a trainer for testing the Nexus Model"""
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
        """Test the train step with Nexus.
        The weights should be updated.
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
        """Test the eval step with NExus.
        The weights should not be updated.
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
        """Test main train loop with NExus"""
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
        """Test checkpoint saving of the Nexus Model"""
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
        """Test the creation of chekpoints during training"""
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"NEXUS_training_{trainer._training_signature}"
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
        """Test the final model saving of the Nexus model and check that we can reload the model
        with AutoModel.
        """
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"NEXUS_training_{trainer._training_signature}"
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
