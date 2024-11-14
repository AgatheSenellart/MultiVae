import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from pythae.models.base.base_model import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.auto_model.auto_model import AutoModel
from multivae.models.jnf_gmc import JNFGMC, JNFGMCConfig
from multivae.models.gmc import GMC,GMCConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP, BaseDictEncoders, BaseDictDecoders, MultipleHeadJointEncoder
from multivae.trainers.multistage import MultistageTrainer, MultistageTrainerConfig

class Test_model:
    
    @pytest.fixture(params=["complete"])
    def dataset(self, request):
        # Create simple small dataset
        data = dict(
            mod1=torch.randn((6, 2)),
            mod2=torch.randn((6, 3)),
            mod3=torch.randn((6, 4)),
            mod4=torch.randn((6, 4)),
        )
        labels = np.array([0, 1, 0, 0, 1,0])
        if request.param == "complete":
            dataset = MultimodalBaseDataset(data, labels)
        else:
            masks = dict(
                mod1=torch.Tensor([True] * 3 + [False] * 3),
                mod2=torch.Tensor([True] * 6),
                mod3=torch.Tensor([True] * 6),
                mod4=torch.Tensor([True] * 6),
            )
            dataset = IncompleteDataset(data=data, masks=masks, labels=labels)

        return dataset



    @pytest.fixture(params=[(1.0, 'standard', True), (1.5,'softplus', False), (2.0, 'softplus', True)])
    def model_config(self, request):
        model_config = JNFGMCConfig(
            n_modalities=4,
            latent_dim=5,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            beta=request.param[0],
            nb_epochs_gmc=10,
            warmup=15,
            logits_to_std = request.param[1],
            add_reconstruction_terms=request.param[2]
        )

        return model_config
    
    @pytest.fixture
    def gmc_config(self):
        return GMCConfig(
            n_modalities=4,
            input_dims=dict(mod1=(2,), mod2=(3,), mod3=(4,), mod4=(4,)),
            common_dim= 10,
            latent_dim = 5,
            temperature=0.2
        )
        
    @pytest.fixture
    def custom_architectures(self, gmc_config):
        # Create an instance of mvae model
        config1 = BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=5)

        encoders = dict(
            mod1=Encoder_VAE_MLP(config1),
            mod2=Encoder_VAE_MLP(config2),
            mod3=Encoder_VAE_MLP(config3),
            mod4=Encoder_VAE_MLP(config3),
        )
        
        config1 = BaseAEConfig(input_dim=(2,), latent_dim=5)
        config2 = BaseAEConfig(input_dim=(3,), latent_dim=5)
        config3 = BaseAEConfig(input_dim=(4,), latent_dim=5)

        decoders = dict(
            mod1=Decoder_AE_MLP(config1),
            mod2=Decoder_AE_MLP(config2),
            mod3=Decoder_AE_MLP(config3),
            mod4=Decoder_AE_MLP(config3),
        )

        return dict(
            encoders=encoders,
            decoders=decoders,
        )
    
    
    @pytest.fixture
    def encoders(self, gmc_config):
        """Basic encoders for each modality"""
        
        return BaseDictEncoders(
            input_dims=gmc_config.input_dims,
            latent_dim=gmc_config.common_dim
        )
        
    
    @pytest.fixture(params = [True, False])
    def joint_encoder(self, gmc_config, request):
        
        config = BaseAEConfig(latent_dim=gmc_config.common_dim)
        if request.param :
            return MultipleHeadJointEncoder(
            BaseDictEncoders(input_dims=gmc_config.input_dims,latent_dim=100), args=config
        )
        else :
            return None
        
    @pytest.fixture
    def shared_encoder(self, gmc_config):
        
        
        return Encoder_VAE_MLP(BaseAEConfig(latent_dim = gmc_config.latent_dim, input_dim = (gmc_config.common_dim,)))
    
    @pytest.fixture
    def gmc_model(self, gmc_config,encoders, joint_encoder, shared_encoder):
        
        return GMC(model_config=gmc_config, processors=encoders, joint_encoder=joint_encoder, shared_encoder=shared_encoder)

    @pytest.fixture(params=[True, False])
    def model(self, custom_architectures, model_config,gmc_model, request):
        custom = request.param
        if custom:
            model = JNFGMC(model_config,gmc_model=gmc_model, **custom_architectures)
        else:
            model = JNFGMC(model_config,gmc_model=gmc_model)
        return model
    
    def test_config(self, model, dataset, model_config,gmc_config):
        assert model.beta == model_config.beta
        assert model.model_config.gmc_config == gmc_config.to_dict()

        

    def test(self, model, dataset, model_config):

        output = model(dataset, epoch=1)
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad
        
        output = model(dataset, epoch=model_config.nb_epochs_gmc+1)
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad
        
        output = model(dataset, epoch=model_config.nb_epochs_gmc+model_config.warmup+1)
        loss = output.loss
        assert type(loss) == torch.Tensor
        assert loss.size() == torch.Size([])
        assert loss.requires_grad

        # Try encoding and prediction
        outputs = model.encode(dataset[0])
        assert outputs.one_latent_space
        embeddings = outputs.z
        assert isinstance(outputs, ModelOutput)
        assert embeddings.shape == (1, 5)
        embeddings = model.encode(dataset[0], N=2).z
        assert embeddings.shape == (2, 1, 5)
        embeddings = model.encode(dataset, cond_mod=["mod2"]).z
        assert embeddings.shape == (len(dataset), 5)
        embeddings = model.encode(dataset, cond_mod="mod3", N=10).z
        assert embeddings.shape == (10, len(dataset), 5)
        embeddings = model.encode(dataset, cond_mod=["mod2", "mod4"]).z
        assert embeddings.shape == (len(dataset), 5)

        Y = model.predict(dataset, cond_mod="mod2")
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset), 2)
        assert Y.mod2.shape == (len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (10, len(dataset), 2)
        assert Y.mod2.shape == (10, len(dataset), 3)

        Y = model.predict(dataset, cond_mod="mod2", N=10, flatten=True)
        assert isinstance(Y, ModelOutput)
        assert Y.mod1.shape == (len(dataset) * 10, 2)
        assert Y.mod2.shape == (len(dataset) * 10, 3)


    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return MultistageTrainerConfig(
            num_epochs=3,
            steps_saving=2,
            learning_rate=1e-4,
            optimizer_cls="AdamW",
            optimizer_params={"betas": (0.91, 0.995)},
            output_dir=dir_path,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4
        )

    @pytest.fixture
    def trainer(self, model, training_config, dataset):
        trainer = MultistageTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
        )

        return trainer

    def new_trainer(self, model, training_config, dataset, checkpoint_dir):
        trainer = MultistageTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=training_config,
            checkpoint=checkpoint_dir,
        )

        return trainer

    def test_train_step(self, trainer):
        
        # test gmc training
        start_gmc_model_state_dict = deepcopy(trainer.model.gmc_model.state_dict())
        start_model_state_dict = deepcopy(trainer.model.state_dict())
        start_optimizer = trainer.optimizer
        _ = trainer.train_step(epoch=1)

        step_1_gmc_model_state_dict = deepcopy(trainer.model.gmc_model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_gmc_model_state_dict[key], step_1_gmc_model_state_dict[key])
                for key in start_gmc_model_state_dict.keys()
            ]
        )
        assert trainer.optimizer == start_optimizer
        
        # after gmc training, check that optimizer has changed and gmc module is not changing
        for epoch in trainer.model.reset_optimizer_epochs :
            
            _ = trainer.prepare_train_step(epoch, None, None)
            
            assert all(
                [
                    torch.equal(trainer.model.state_dict()[key], trainer._best_model.state_dict()[key])
                    for key in trainer.model.state_dict().keys()
                ]
            )
            
            step_2_gmc_model_state_dict = deepcopy(trainer.model.gmc_model.state_dict())
            step_2_model_state_dict = deepcopy(trainer.model.state_dict())
            
            _ = trainer.train_step(epoch=epoch)

            step_3_gmc_model_state_dict = deepcopy(trainer.model.gmc_model.state_dict())
            step_3_model_state_dict = deepcopy(trainer.model.state_dict())
            
            # check that weights were not updated for gmc
            assert all(
                [
                    torch.equal(step_2_gmc_model_state_dict[key], step_3_gmc_model_state_dict[key])
                    for key in step_2_gmc_model_state_dict.keys()
                ]
            )
            assert trainer.optimizer != start_optimizer
            # check that weights were updated in the rest of the model
            assert not all(
                [
                    torch.equal(step_2_model_state_dict[key], step_3_model_state_dict[key])
                    for key in step_2_model_state_dict.keys()
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

    def test_checkpoint_saving_during_training(
        self, model, trainer, training_config, dataset
    ):
        #
        target_saving_epoch = training_config.steps_saving

        dir_path = training_config.output_dir

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"JNFGMC_training_{trainer._training_signature}"
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
            dir_path, f"JNFGMC_training_{trainer._training_signature}"
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
            dir_path, f"JNFGMC_training_{trainer._training_signature}"
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
#         if not hasattr(dataset, "masks"):
#             nll = model.compute_joint_nll(dataset, K=10, batch_size_K=2)
#             assert nll >= 0
#             assert type(nll) == torch.Tensor
#             assert nll.size() == torch.Size([])

#             cnll = model.cond_nll_from_subset(
#                 dataset, ["mod1", "mod2"], ["mod3"], K=10, batch_size_k=2
#             )
#             assert type(cnll) == dict
#             assert "mod3" in cnll.keys()
