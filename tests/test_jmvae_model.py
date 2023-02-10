import os
import numpy as np
import pytest

from multivae.data.datasets.base import MultimodalBaseDataset
import os
import numpy as np
import pytest
from copy import deepcopy
from torch import nn
import torch

from multivae.models.base import BaseMultiVAE, BaseMultiVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.models.jmvae import JMVAE, JMVAEConfig
from multivae.models.joint_models import BaseJointModel
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.nn.benchmarks.mnist.convnets import Encoder_Conv_AE_MNIST, Decoder_Conv_AE_MNIST
from pythae.models.base import BaseAEConfig


class Test:
    @pytest.fixture
    def input1(self):
        
        # Create simple small dataset
        data = dict(
            mod1 = torch.Tensor([[1.0,2.0],[4.0,5.0]]),
            mod2 = torch.Tensor([[67.1,2.3,3.0],[1.3,2.,3.]]),
        )
        labels = np.array([0,1])
        dataset = MultimodalBaseDataset(data, labels)
        
        # Create an instance of jmvae model
        model_config = JMVAEConfig(n_modalities=2, latent_dim=5)
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)

        encoders = dict(
            mod1 = Encoder_VAE_MLP(config1),
            mod2 = Encoder_VAE_MLP(config2)
        )
        
        decoders = dict(
            mod1 = Decoder_AE_MLP(config1),
            mod2 = Decoder_AE_MLP(config2)
        )
        
        return dict(model_config = model_config,
                    encoders = encoders,
                    decoders = decoders,
                    dataset = dataset)

    def test1(self, input1):
        model = JMVAE(**input1)

        assert model.alpha == input1['model_config'].alpha
        
        loss = model(input1['dataset'], epoch=2, warmup=2).loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        
    @pytest.fixture
    def input2(self):
        
        # Create simple small dataset
        data = dict(
            mod1 = torch.Tensor([[1.0,2.0],[4.0,5.0]]),
            mod2 = torch.Tensor([[67.1,2.3,3.0],[1.3,2.,3.]]),
        )
        labels = np.array([0,1])
        dataset = MultimodalBaseDataset(data, labels)
        
        # Create an instance of jmvae model
        model_config = JMVAEConfig(n_modalities=2, latent_dim=5, input_dims=dict(mod1=(2,), mod2=(3,)))
        
        
        return dict(model_config = model_config,
                    dataset = dataset)

    def test2(self, input2):
        model = JMVAE(**input2)

        assert model.alpha == input2['model_config'].alpha
        
        loss = model(input2['dataset'], epoch=2, warmup=2).loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        

class TestTraining:
    @pytest.fixture
    def input_dataset(self):
        
        # Create simple small dataset
        data = dict(
            mod1 = torch.Tensor([[1.0,2.0],[4.0,5.0]]),
            mod2 = torch.Tensor([[67.1,2.3,3.0],[1.3,2.,3.]]),
        )
        labels = np.array([0,1])
        dataset = MultimodalBaseDataset(data, labels)
        
        
        return dataset

    @pytest.fixture
    def model_config(self, input_dataset):
        return JMVAEConfig(
            n_modalities=int(len(input_dataset.data.keys())),
            latent_dim=5,
            input_dims=dict(
                mod1=tuple(input_dataset[0].data["mod1"].shape),
                mod2=tuple(input_dataset[0].data["mod2"].shape)
                ))

    @pytest.fixture
    def custom_architecture(self):
        config = BaseAEConfig(input_dim=(10,2), latent_dim=10)
        encoders = dict(
            mod1 = Encoder_VAE_MLP(config),
            mod2 = Encoder_Conv_AE_MNIST(config)
        )
        decoders = dict(
            mod1 = Decoder_AE_MLP(config),
            mod2 = Decoder_Conv_AE_MNIST(config)
        )
        return {"encoders": encoders, "decoders": decoders}
    
    @pytest.fixture(
        params=[
            True,
            False,
        ]
    )
    def model(self, model_config, custom_architecture, request):
        # randomized

        custom = request.param

        if not custom:
            model = JMVAE(model_config)

        else:
            model = JMVAE(
                model_config,
                encoder=custom_architecture['encoders'],
                decoder=custom_architecture['decoders']
            )

        return model

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
            output_dir=dir_path
        )

    @pytest.fixture
    def trainer(self, model, training_config, input_dataset):
        

        trainer = BaseTrainer(
            model=model,
            train_dataset=input_dataset,
            eval_dataset=input_dataset,
            training_config=training_config
        )

        trainer.prepare_training()

        return trainer


    def test_train_step(self, trainer):

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

        # check pickled custom decoder
        if not model.model_config.uses_default_decoders:
            assert "decoders.pkl" in files_list

        else:
            assert not "decoders.pkl" in files_list

        # check pickled custom encoder
        if not model.model_config.uses_default_encoders:
            assert "encoders.pkl" in files_list

        else:
            assert not "encoders.pkl" in files_list

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
        model_rec = JMVAE.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())

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
        #
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

        # check pickled custom decoder
        if not model.model_config.uses_default_decoders:
            assert "decoders.pkl" in files_list

        else:
            assert not "decoders.pkl" in files_list

        # check pickled custom encoder
        if not model.model_config.uses_default_encoders:
            assert "encoders.pkl" in files_list

        else:
            assert not "encoders.pkl" in files_list

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

        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"JMVAE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not model.model_config.uses_default_decoders:
            assert "decoders.pkl" in files_list

        else:
            assert not "decoders.pkl" in files_list

        # check pickled custom encoder
        if not model.model_config.uses_default_encoders:
            assert "encoders.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check reload full model
        model_rec = JMVAE.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())


