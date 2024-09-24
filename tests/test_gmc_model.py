from multivae.models.gmc import GMC, GMCConfig
import pytest
from multivae.data.datasets import MultimodalBaseDataset
import torch 
import numpy as np
from multivae.models.nn.default_architectures import BaseDictEncoders, MultipleHeadJointEncoder, Encoder_VAE_MLP, ModelOutput, BaseAEConfig
import os
from copy import deepcopy
from multivae.models import AutoModel
from multivae.trainers import BaseTrainer, BaseTrainerConfig


class Test:
    
    @pytest.fixture(params=[2,3])
    def n_modalities(self, request):
        return request.param
    
    @pytest.fixture(params=[(3,28,28),(122,)])
    def input_dims(self, request,n_modalities):
        d = {f'm{i}' : request.param for i in range(n_modalities)}
        return d
    
    
    @pytest.fixture
    def dataset(self, input_dims):
        
        data = dict()
        for m in input_dims:
            data[m] = torch.from_numpy(np.random.randn(100,*input_dims[m])).float()

            
        return MultimodalBaseDataset(data=data)
    
    @pytest.fixture(params=["between_modality_pairs","between_modality_joint"])
    def model_config(self, n_modalities, input_dims,request):
        return GMCConfig(
            n_modalities=n_modalities,
            input_dims=input_dims,
            common_dim= 10,
            latent_dim = 5,
            temperature=0.2,
            loss=request.param
        )
    
    
    @pytest.fixture
    def encoders(self, input_dims, model_config):
        """Basic encoders for each modality"""
        
        return BaseDictEncoders(
            input_dims=input_dims,
            latent_dim=model_config.common_dim
        )
        
    
    @pytest.fixture(params = [True, False])
    def joint_encoder(self, input_dims, model_config, request):
        
        config = BaseAEConfig(latent_dim=model_config.common_dim)
        if request.param:
            return MultipleHeadJointEncoder(
            BaseDictEncoders(input_dims=input_dims,latent_dim=100), args=config
        )
        else :
            return None
        
    @pytest.fixture
    def shared_encoder(self, model_config):
        
        
        return Encoder_VAE_MLP(BaseAEConfig(latent_dim = model_config.latent_dim, input_dim = (model_config.common_dim,)))
    
    @pytest.fixture
    def model(self, model_config,encoders, joint_encoder, shared_encoder):
        
        return GMC(model_config=model_config, processors=encoders, joint_encoder=joint_encoder, shared_encoder=shared_encoder)
    
    
    def test_model_setup(self, model, model_config,encoders, joint_encoder,shared_encoder):
        
        if model_config.loss == "between_modality_pairs" or joint_encoder is None:
            assert model.loss == "pairs"
            assert "joint_encoder" not in model.model_config.custom_architectures
        else :
            assert model.loss == "joint"
            assert "joint_encoder" in model.model_config.custom_architectures
            
        assert "shared_encoder" in model.model_config.custom_architectures
        assert "processors" in model.model_config.custom_architectures
    
    
    def test_forward(self, model, dataset):
        
        output = model(dataset)
        assert isinstance(output, ModelOutput)
        
        loss = output.loss
        
        assert isinstance(loss, torch.Tensor)
        
    
    def test_encode(self, model, dataset):
        
        cond_mod = 'all'
        
        if model.loss == 'pairs':
            with pytest.raises(AttributeError):
                model.encode(dataset,cond_mod = cond_mod)
        
        else :
            output = model.encode(dataset,cond_mod = cond_mod)
            assert isinstance(output, ModelOutput)
            embedding = output.embedding
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape == (len(dataset),model.latent_dim)
            
        cond_mod = 'm1'
    
        output = model.encode(dataset,cond_mod = cond_mod)
        assert isinstance(output, ModelOutput)
        embedding = output.embedding
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (len(dataset),model.latent_dim)
        
        
        
    def test_save_load(self, model, dataset, tmpdir):
        
        # perform one step optimization
        optimizer = torch.optim.Adam(model.parameters())
    
        optimizer.zero_grad()
        
        loss = model(dataset).loss
        
        loss.backward()
        
        optimizer.step()
        
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        
        model_before_save = deepcopy(model)
        
        model.save(dir_path)
        
        model_after_save = GMC.load_from_folder(dir_path)
        
        assert all(
            [
                torch.equal(
                    model_after_save.state_dict()[key].cpu(), model_before_save.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )
        
        # Test reload with AutoModel
        model_after_save = AutoModel.load_from_folder(dir_path)
        
        assert all(
            [
                torch.equal(
                    model_after_save.state_dict()[key].cpu(), model_before_save.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )
        
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

        return trainer

    def new_trainer(self, model, training_config, dataset, checkpoint_dir):
        trainer = BaseTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
            checkpoint=checkpoint_dir,
        )

        return trainer

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

        assert set(
            ["model.pt", "optimizer.pt", "training_config.json", "info_checkpoint.json"]
        ).issubset(set(files_list))

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

        assert type(model_rec.processors.cpu()) == type(model.processors.cpu())

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

    def test_checkpoint_saving_during_training(
        self, model, trainer, training_config, dataset
    ):
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"GMC_training_{trainer._training_signature}"
        )

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        # try resuming
        new_trainer_ = self.new_trainer(model, training_config, dataset, checkpoint_dir)

        assert new_trainer_.best_train_loss == trainer.best_train_loss
        assert new_trainer_.trained_epochs == target_saving_epoch

        new_trainer_.train()

    def test_resume_from_checkpoint(self, model, trainer, training_config):
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"GMC_training_{trainer._training_signature}"
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
        dir_path = training_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"GMC_training_{trainer._training_signature}"
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
        
        
        
        
        