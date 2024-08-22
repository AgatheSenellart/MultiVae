import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from encoders import Decoder_test, Encoder_test
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.benchmarks.mnist.convnets import (
    Decoder_Conv_AE_MNIST,
    Encoder_Conv_AE_MNIST,
)
from pythae.models.normalizing_flows import IAF, IAFConfig
from torch import nn

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models import AutoModel, Nexus, NexusConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class Test:
    @pytest.fixture(params=["complete", "incomplete"])
    def dataset(self, request):
        # Create simple small dataset
        data = dict(
            mod1=torch.Tensor(np.random.random((4, 1, 28, 28))),
            mod2=torch.Tensor(np.random.random((4, 3, 28, 28))),
            mod3=torch.Tensor(np.random.random((4, 3, 32, 32))),
            mod4=torch.Tensor(np.random.random((4, 1, 32, 23))),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
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
                    mod1=(1, 28, 28),
                    mod2=(3, 28, 28),
                    mod3=(3, 32, 32),
                    mod4=(1, 32, 23),
                ),
                modalities_specific_dim=dict(mod1=3, mod2=4, mod3=3, mod4=4),
                gammas=dict(mod1=1.2, mod2=1.3, mod3=1.0, mod4=3.0),
                bottom_betas=dict(mod1=2.0, mod2=1.0, mod3=1.4, mod4=0.4),
                msg_dim=11,
                latent_dim=4,
                top_beta=3,
            ),
        ]
    )
    def custom_config_archi(self, request):
        encoders = dict()
        decoders = dict()
        top_encoders = dict()
        top_decoders = dict()
        for m in request.param["modalities_specific_dim"]:
            encoders[m] = Encoder_test(
                BaseAEConfig(
                    input_dim=request.param["input_dims"][m],
                    latent_dim=request.param["modalities_specific_dim"][m],
                )
            )

            decoders[m] = Decoder_test(
                BaseAEConfig(
                    input_dim=request.param["input_dims"][m],
                    latent_dim=request.param["modalities_specific_dim"][m],
                )
            )

            top_encoders[m] = Encoder_test(
                BaseAEConfig(
                    input_dim=(request.param["modalities_specific_dim"][m],),
                    latent_dim=request.param["msg_dim"],
                )
            )

            top_decoders[m] = Decoder_test(
                BaseAEConfig(
                    input_dim=(request.param["modalities_specific_dim"][m],),
                    latent_dim=request.param["latent_dim"],
                )
            )

        joint_encoder = Encoder_test(
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

    @pytest.fixture(params=["custom_architectures", "default_architectures"])
    def model(self, custom_config_archi, request):
        if request.param == "custom_architectures":
            return Nexus(**custom_config_archi)
        else:
            return Nexus(model_config=custom_config_archi["model_config"])

    def test_forward(self, model, dataset):
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
        assert embeddings.shape == (1, model.latent_dim)
        embeddings = model.encode(dataset[0], N=2).z
        assert embeddings.shape == (2, 1, model.latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod2"]).z
        assert embeddings.shape == (4, model.latent_dim)
        embeddings = model.encode(dataset, cond_mod="mod3", N=10).z
        assert embeddings.shape == (10, 4, model.latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod2", "mod4"]).z
        assert embeddings.shape == (4, model.latent_dim)

        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (4, 1, 28, 28)
        assert Y.mod2.shape == (4, 3, 28, 28)

        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, 4, 1, 28, 28)
        assert Y.mod2.shape == (10, 4, 3, 28, 28)

        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (4 * 10, 1, 28, 28)
        assert Y.mod2.shape == (4 * 10, 3, 28, 28)

    def test_grad_with_missing_inputs(self, model, dataset):
        # check that the grad with regard to missing modality is 0

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

        trainer.prepare_training()

        return trainer

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_checkpoint_saving_during_training(self, model, trainer, training_config):
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
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"NEXUS_training_{trainer._training_signature}"
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


#     def test_compute_nll(self, model, dataset):
#         nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
#         assert nll >= 0
#         assert type(nll) == torch.Tensor
#         assert nll.size() == torch.Size([])
