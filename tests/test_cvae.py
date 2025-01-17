from pytest import fixture, mark
import pytest
from multivae.data import MultimodalBaseDataset
from multivae.models.cvae import CVAE, CVAEConfig
import torch
from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    Encoder_VAE_MLP,
    MultipleHeadJointEncoder,
    ConditionalDecoder_MLP,
    ModelOutput,
    BaseDictEncoders)
from multivae.models.nn.mmnist import EncoderConvMMNIST
from multivae.trainers import BaseTrainer, BaseTrainerConfig
import tempfile
from copy import deepcopy
import os
from multivae.models.auto_model import AutoModel

class Test_CVAE:
    
    @fixture
    def dataset(self):
        return MultimodalBaseDataset(
            data=dict(
                mnist=torch.randn((100,3,28,28)),
                label=torch.randint(10,(100,10)).float()
            )
        )
    
    @fixture(params=[
        [10,'normal',{}],
        [14,'laplace',dict(scale=0.5)]
    ])
    def model_config(self, request):
        
        return CVAEConfig(
            input_dims = dict(mnist= (3,28,28), label=(10,)),
            latent_dim=request.param[0],
            conditioning_modality = 'label',
            main_modality='mnist',
            decoder_dist =request.param[1],
            decoder_dist_params=request.param[2]
        )
        
    @fixture(params=[ True, False
        
    ])
    def architectures(self, model_config,request):
        
        if request.param:
            prior_network = Encoder_VAE_MLP(BaseAEConfig(input_dim=(10,), latent_dim=model_config.latent_dim))
            encoder = MultipleHeadJointEncoder(
            dict_encoders=dict(mnist = EncoderConvMMNIST(BaseAEConfig(input_dim=(3,28,28), latent_dim=model_config.latent_dim)),
                               label= Encoder_VAE_MLP(BaseAEConfig(input_dim=(10,), latent_dim=model_config.latent_dim))),
            args=model_config,
            n_hidden_layers=1,
            hidden_dim=128
        )
            decoder = ConditionalDecoder_MLP(model_config.latent_dim,(10,), (3,28,28))
        else:
            prior_network = None
            encoder=None
            decoder=None
            
            
        return dict(encoder = encoder, decoder= decoder,prior_network= prior_network)

    def test_setup(self, model_config, architectures):
        
        model = CVAE(model_config=model_config,
                     **architectures)
        
        assert model.latent_dim == model_config.latent_dim
        assert model.conditioning_modality == model_config.conditioning_modality
        assert model.main_modality == model_config.main_modality
        
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'prior_network')
        
        if architectures['encoder'] is not None:
            assert model.encoder == architectures['encoder']
        else:
            assert isinstance(model.encoder, MultipleHeadJointEncoder)
        
        if architectures['decoder'] is not None:
            assert model.decoder == architectures['decoder']
        else:
            assert isinstance(model.decoder, ConditionalDecoder_MLP)
        
        if architectures['prior_network'] is not None:
            assert model.prior_network == architectures['prior_network']
        else:
            assert model.prior_network is None
            
            
        
        return 
    
    @fixture
    def model(self, model_config,architectures):
        return CVAE(model_config=model_config,
                     **architectures)
        
    
    def test_forward(self, model, dataset):
        
        samples = dataset[:10]
        output = model(samples)
        
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'loss')
        assert isinstance(output.loss, torch.Tensor)
        
        return
    
    def test_encode(self, dataset, model):
        
        samples = dataset[:10]
        output = model.encode(samples)
        
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert torch.all(output.cond_mod_data == samples.data['label'])
        
        output = model.encode(samples, N=4)
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(4,10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert output.cond_mod_data.shape == (4,*samples.data['label'].shape)
        
        output = model.encode(samples, N=4, flatten=True)
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(4*10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert torch.all(output.cond_mod_data == torch.cat([samples.data['label']]*4))
        
        
        return
    
    def test_decode(self, model, dataset):
        
        samples=dataset[:10]
        
        embeddings=model.encode(samples)
        output = model.decode(embeddings)
        
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == samples.data['mnist'].shape
        
        
        
        return 
    
    
    def test_generate_from_prior(self, model, dataset):
        
        samples = dataset[:10]
        
        output = model.generate_from_prior(cond_mod_data=samples.data['label'])
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert torch.all(output.cond_mod_data == samples.data['label'])
        
        output = model.generate_from_prior(cond_mod_data=samples.data['label'], N=4)
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(4,10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert output.cond_mod_data.shape == (4,*samples.data['label'].shape)
        
        output = model.encode(samples, N=4, flatten=True)
        assert isinstance(output, ModelOutput)
        assert output.z.shape==(4*10,model.latent_dim)
        assert hasattr(output, 'cond_mod_data')
        assert torch.all(output.cond_mod_data == torch.cat([samples.data['label']]*4))
        
        return
        
            
    
    def test_predict(self, model, dataset):
        
        samples = dataset[:10]
        
        # Test reconstruction
        output=model.predict(cond_mod = 'mnist', gen_mod = 'mnist',inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == samples.data['mnist'].shape
        
        output=model.predict(cond_mod = 'mnist', gen_mod = 'mnist',inputs=samples, N=10)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == (10,10,3,28,28)
        
        output=model.predict(cond_mod = ['mnist'], gen_mod = 'mnist',inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == samples.data['mnist'].shape
        
        output=model.predict(cond_mod = ['mnist','label'], gen_mod = 'mnist',inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == samples.data['mnist'].shape
        
        # Test generation
        output=model.predict(cond_mod = 'label', gen_mod = 'mnist',inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, 'reconstruction')
        assert output.reconstruction.shape == samples.data['mnist'].shape
        
        # Test that issues are raised when the input is not correct
        with pytest.raises(ValueError):
            output=model.predict(cond_mod = 'svhn', gen_mod = 'mnist',inputs=samples)
        with pytest.raises(ValueError):
            output=model.predict(cond_mod = ['svhn','label'], gen_mod = 'mnist',inputs=samples)
        with pytest.raises(ValueError):
            output=model.predict(cond_mod = ['svhn','label','mnist'], gen_mod = 'mnist',inputs=samples)

        
        return
    
    
    @fixture(params = [
        [32,64, 3,'Adagrad'],
        [16,16,4,'Adam']
    ])
    def trainer_config(self, request):
        
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
    def trainer(self, trainer_config, model,dataset):
        
        return BaseTrainer(model, train_dataset=dataset,eval_dataset=dataset
                              ,training_config=trainer_config)
        
    @mark.slow
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
    
    @mark.slow   
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
    @mark.slow
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
    @mark.slow    
    def test_checkpoint_saving(self, model, trainer, trainer_config):
        dir_path = trainer_config.output_dir

        # Make a training step, save the model and reload it
        step_1_loss = trainer.train_step(epoch=1)

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
    @mark.slow 
    def test_checkpoint_saving_during_training(self, model, trainer, trainer_config):
        
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
        dir_path = trainer_config.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"CVAE_training_{trainer._training_signature}"
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

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        
        