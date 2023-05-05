import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from encoders import Encoder_test
from pytest import fixture
from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows import IAF, IAFConfig

from multivae.data.datasets import MultimodalBaseDataset
from multivae.models.auto_model.auto_model import AutoModel
from multivae.models.dcca import DCCA, DCCAConfig
from multivae.models.jnf_dcca import JNFDcca, JNFDccaConfig
from multivae.models.nn.default_architectures import BaseDictEncoders, Decoder_AE_MLP
from multivae.trainers.add_dcca_trainer import AddDccaTrainer, AddDccaTrainerConfig


class TestDcca:
    @fixture(
        params=[
            dict(n_modalities=2, embedding_dim=5, use_all_singular_values=False),
            dict(n_modalities=5, embedding_dim=10, use_all_singular_values=False),
        ]
    )
    def inputs(self, request):
        n_mod = request.param["n_modalities"]
        embed_dim = request.param["embedding_dim"]
        networks = BaseDictEncoders(
            {"mod" + str(k): (k,) for k in range(1, n_mod + 1)}, embed_dim
        )

        data = {"mod" + str(k): torch.ones((10, k)) for k in range(1, n_mod + 1)}
        return networks, DCCAConfig(**request.param), MultimodalBaseDataset(data)

    def test(self, inputs):
        networks, config, data = inputs
        model = DCCA(config, networks)

        assert model.latent_dim == config.embedding_dim
        assert model.use_all_singular_values == config.use_all_singular_values

        output = model(data)
        assert hasattr(output, "loss")

    def test_raises_error_forward(self, inputs):
        networks, config, data = inputs
        model = DCCA(config, networks)

        with pytest.raises(AttributeError):
            model(MultimodalBaseDataset({"unknown_modality": 10}))

    def test_set_networks(self, inputs):
        with pytest.raises(AssertionError):
            _ = DCCA(inputs[1], {"mod1": AutoModel()})

        inputs[0]["mod1"].latent_dim = inputs[1].embedding_dim + 1

        with pytest.raises(AttributeError):
            _ = DCCA(inputs[1], inputs[0])


class TestJNFDcca:
    @fixture
    def dataset(self):
        # Create simple small dataset with 2 modalities
        data = dict(
            mod1=torch.rand((200, 2)),
            mod2=torch.rand((200, 3)),
            mod3=torch.rand((200, 4)),
        )
        labels = np.random.randint(2, size=200)
        dataset = MultimodalBaseDataset(data, labels)
        return dataset

    @fixture
    def custom_architectures(self):
        # Create custom instances for the dcca_networks and decoders
        config_dcca_1 = BaseAEConfig(input_dim=(2,), latent_dim=2)
        config_dcca_2 = BaseAEConfig(input_dim=(3,), latent_dim=2)
        config_dcca_3 = BaseAEConfig(input_dim=(4,), latent_dim=2)

        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        dcca_networks = dict(
            mod1=Encoder_test(config_dcca_1),
            mod2=Encoder_test(config_dcca_2),
            mod3=Encoder_test(config_dcca_3),
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
            dcca_networks=dcca_networks,
            decoders=decoders,
            flows=flows,
        )

    @fixture(params=[False])
    def model_config(self, request):
        model_config = JNFDccaConfig(
            n_modalities=3,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,)),
            use_all_singular_values=request.param,
            embedding_dcca_dim=2,
            nb_epochs_dcca=2,
            warmup=2,
        )

        return model_config

    @fixture(params=[True, False])
    def model(self, custom_architectures, model_config, request):
        custom = request.param
        if custom:
            model = JNFDcca(model_config, **custom_architectures)
        else:
            model = JNFDcca(model_config)

        return model

    @fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return AddDccaTrainerConfig(
            num_epochs=5,
            steps_saving=2,
            learning_rate=1e-3,
            optimizer_cls="AdamW",
            optimizer_params={"betas": (0.91, 0.995)},
            output_dir=dir_path,
            learning_rate_dcca=1e-4,
        )

    @fixture
    def trainer(self, model, training_config, dataset):
        trainer = AddDccaTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        trainer.prepare_training()

        return trainer

    def test_model_forward(self, model, dataset, model_config):
        assert hasattr(model, "dcca_networks")
        assert model.warmup == model_config.warmup
        assert model.nb_epochs_dcca == model_config.nb_epochs_dcca

        # Test forward method during dcca training
        output = model(dataset, epoch=1)
        assert not hasattr(output, "recon_loss")
        assert not hasattr(output, "KLD")
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # Test forward method during joint vae training
        output = model(dataset, epoch=model_config.nb_epochs_dcca + 1)
        assert hasattr(output, "recon_loss")
        assert hasattr(output, "KLD")
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        # assert loss.requires_grad

        # Test forward method during flows training
        output = model(
            dataset, epoch=model_config.nb_epochs_dcca + model_config.warmup + 2
        )
        assert hasattr(output, "ljm")
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # Try encoding and prediction
        outputs = model.encode(dataset)
        assert outputs.one_latent_space
        embeddings = outputs.z
        assert isinstance(outputs, ModelOutput)
        assert embeddings.shape == (200, model_config.latent_dim)
        embeddings = model.encode(dataset, N=2).z
        assert embeddings.shape == (2, 200, model_config.latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod1"]).z
        assert embeddings.shape == (200, model_config.latent_dim)
        embeddings = model.encode(dataset, cond_mod="mod2", N=10).z
        assert embeddings.shape == (10, 200, model_config.latent_dim)
        embeddings = model.encode(dataset, cond_mod=["mod2", "mod1"], mcmc_steps=2).z
        assert embeddings.shape == (200, model_config.latent_dim)

        Y = model.predict(dataset, cond_mod="mod1")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (200, 2)
        assert Y.mod2.shape == (200, 3)

        Y = model.predict(dataset, cond_mod="mod1", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, 200, 2)
        assert Y.mod2.shape == (10, 200, 3)

        Y = model.predict(dataset, cond_mod="mod1", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (200 * 10, 2)
        assert Y.mod2.shape == (200 * 10, 3)

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
        assert trainer.training_config.learning_rate == 1e-4
        _ = trainer.prepare_train_step(trainer.model.nb_epochs_dcca + 1, None, None)
        _ = trainer.train_step(epoch=trainer.model.nb_epochs_dcca + 1)
        assert trainer.training_config.learning_rate == 1e-3
        step_2_model_state_dict = deepcopy(trainer.model.state_dict())

        assert not all(
            [
                torch.equal(step_1_model_state_dict[key], step_2_model_state_dict[key])
                for key in step_1_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer != start_optimizer

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
        assert hasattr(trainer.model, "dcca_networks")
        dir_path = training_config.output_dir

        # Make a training step
        step_1_loss = trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)

        assert hasattr(trainer.model, "dcca_networks")

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
            dir_path, f"JNFDcca_training_{trainer._training_signature}"
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
            dir_path, f"JNFDcca_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

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
        assert type(model_rec.DCCA_module.networks.cpu()) == type(
            model.DCCA_module.networks.cpu()
        )

    def test_compute_nll(self, model, dataset):
        nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
        assert nll >= 0
        assert type(nll) == torch.Tensor
        assert nll.size() == torch.Size([])
