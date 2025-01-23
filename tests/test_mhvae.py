import os
import tempfile
from copy import deepcopy

import pytest
import torch
from pytest import fixture, mark

from multivae.data import MultimodalBaseDataset
from multivae.models.auto_model import AutoModel
from multivae.models.mhvae import MHVAE, MHVAEConfig
from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    BaseDictEncoders,
    ConditionalDecoder_MLP,
    Encoder_VAE_MLP,
    ModelOutput,
    MultipleHeadJointEncoder
)
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.models.base import BaseEncoder,ModelOutput, BaseDecoder
from torch import nn

# Architectures for testing

class my_input_encoder(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.act_1 = nn.SiLU()
        
    def forward(self, x):
       
        x = self.conv0(x)
        x = self.act_1(x)
        
        return ModelOutput(embedding = x)


        
class bu_2(BaseEncoder):
    
    def __init__(self, inchannels,outchannels,latent_dim):
        super().__init__()

        self.network = nn.Sequential( nn.Conv2d(
            in_channels=inchannels,
            out_channels=outchannels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        ) ,
        nn.SiLU(),
        nn.Flatten(),
        nn.Linear(2048, 512),  
        nn.ReLU())
        
        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        h = self.network(x)
        return ModelOutput(
            embedding = self.mu(h),
            log_covariance = self.log_var(h)
        )
        
# Defininin top-down blocks and decoder
        
class td_2(nn.Module):
    
    def __init__(self, latent_dim):
        super().__init__()

        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim,2048),nn.ReLU()
        )
        self.convs = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,bias=True),
                                   nn.SiLU())
    def forward(self,x):
        h=self.linear(x)
        h = h.view(h.shape[0],128,4,4)
        return self.convs(h)
    
class td_1(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1, output_padding=1,bias=True),
                                   nn.SiLU()
            )
    def forward(self, x):
        return self.network(x)

class bu_1(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.network = nn.Sequential(nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True
        ) , nn.SiLU())
    def forward(self, x):
        return self.network(x)

class add_bu(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.network = nn.Sequential(nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        ) , nn.SiLU())
    def forward(self, x):
        return self.network(x)

class my_input_decoder(BaseDecoder):
    
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(32,3,3,2,1, output_padding=1),nn.Sigmoid()
        )
    
    def forward(self,x):
        return ModelOutput(
            reconstruction = self.network(x)
        )
        
# Defining prior blocks and posterior blocks

class prior_block(BaseEncoder):
    
    def __init__(self, n_channels):
        super().__init__()

        self.mu = nn.utils.weight_norm(nn.Conv2d(n_channels,n_channels,1,1,0))
        self.logvar = nn.utils.weight_norm(nn.Conv2d(n_channels,n_channels,1,1,0))
    def forward(self, x):
        return ModelOutput(embedding = self.mu(x), log_covariance = self.logvar(x))

class posterior_block(BaseEncoder):
    
    def __init__(self, n_channels_before_concat):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2*n_channels_before_concat,n_channels_before_concat,3,1,1, bias=True), 
            nn.SiLU()
        )
        
        self.mu = nn.utils.weight_norm(nn.Conv2d(n_channels_before_concat,n_channels_before_concat,1,1,0))
        self.logvar = nn.utils.weight_norm(nn.Conv2d(n_channels_before_concat,n_channels_before_concat,1,1,0))
        
    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding = self.mu(h), log_covariance = self.logvar(h))
    


class Test_CVAE:

    @fixture
    def dataset(self):
        return MultimodalBaseDataset(
            data=dict(
                m1 = torch.randn((100, 3, 28, 28)),
                m0 = torch.randn((100, 3, 28, 28))
            )
        )

    @fixture(params=[[("normal", "normal"), 3, 1.0, 10], [("normal", "laplace"), 4, 2.5, 15]])
    def model_config(self, request):

        return MHVAEConfig(
            n_modalities=2,
            latent_dim=request.param[3],
            decoders_dist=dict(m0 = request.param[0][0], m1=request.param[0][1]),
            n_latent=request.param[1],
            beta=request.param[2]
        )
    



    @fixture
    def architectures(self, model_config):

        encoders = dict(m0 = my_input_encoder(), m1=my_input_encoder())
        decoders = dict(m0 = my_input_decoder(), m1=my_input_decoder())
        bottom_up_blocks = dict(m0 = [bu_1()], m1 = [bu_1()])
        
        if model_config.n_latent == 4:
            bottom_up_blocks['m0'].append(add_bu())
            bottom_up_blocks['m1'].append(add_bu())
            
        bottom_up_blocks['m0'].append(bu_2(64,128,model_config.latent_dim))
        bottom_up_blocks['m1'].append(bu_2(64,128,model_config.latent_dim))
        
        td_blocks = [td_1()]
        if model_config.n_latent == 4:
            td_blocks.append(add_bu())
        
        td_blocks.append(td_2(model_config.latent_dim))
        
        if model_config.n_latent == 4:
            prior_blocks = [prior_block(32), prior_block(64), prior_block(64)]
            posterior_blocks = [posterior_block(32), posterior_block(64), posterior_block(64)]
            
        else:
            prior_blocks = [prior_block(32), prior_block(64)]
            posterior_blocks = [posterior_block(32), posterior_block(64)]
        
        return dict(encoders=encoders, 
                    decoders=decoders,
                    bottom_up_blocks=bottom_up_blocks,
                    top_down_blocks=td_blocks,
                    prior_blocks=prior_blocks,
                    posterior_blocks=posterior_blocks)
        


    def test_setup(self, model_config, architectures):

        model = MHVAE(model_config=model_config, **architectures)

        assert model.latent_dim == model_config.latent_dim
        assert model.beta == model_config.beta
        assert model.n_latent == model_config.n_latent

        assert isinstance(model.encoders, nn.ModuleDict)
        assert isinstance(model.decoders, nn.ModuleDict)
        assert isinstance(model.bottom_up_blocks, nn.ModuleDict)
        assert isinstance(model.top_down_blocks, nn.ModuleList)
        assert isinstance(model.prior_blocks, nn.ModuleList)
        assert isinstance(model.posterior_blocks, nn.ModuleList)



        return

    @fixture
    def model(self, model_config, architectures):
        return MHVAE(model_config=model_config, **architectures)

    def test_forward(self, model, dataset):

        samples = dataset[:10]
        output = model(samples)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "loss")
        assert isinstance(output.loss, torch.Tensor)

        return

    def test_encode(self, dataset, model):

        samples = dataset[:10]
        output = model.encode(samples)

        assert isinstance(output, ModelOutput)
        assert output.z.shape == (10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert torch.all(output.cond_mod_data == samples.data["label"])

        output = model.encode(samples, N=4)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4, 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert output.cond_mod_data.shape == (4, *samples.data["label"].shape)

        output = model.encode(samples, N=4, flatten=True)
        assert isinstance(output, ModelOutput)
        assert output.z.shape == (4 * 10, model.latent_dim)
        assert hasattr(output, "cond_mod_data")
        assert torch.all(output.cond_mod_data == torch.cat([samples.data["label"]] * 4))

        return

    def test_decode(self, model, dataset):

        samples = dataset[:10]

        embeddings = model.encode(samples)
        output = model.decode(embeddings)

        assert isinstance(output, ModelOutput)
        assert hasattr(output, "reconstruction")
        assert output.reconstruction.shape == samples.data["m0"].shape

        return

    # def test_generate_from_prior(self, model, dataset):

    #     samples = dataset[:10]

    #     output = model.generate_from_prior(cond_mod_data=samples.data["label"])
    #     assert isinstance(output, ModelOutput)
    #     assert output.z.shape == (10, model.latent_dim)
    #     assert hasattr(output, "cond_mod_data")
    #     assert torch.all(output.cond_mod_data == samples.data["label"])

    #     output = model.generate_from_prior(cond_mod_data=samples.data["label"], N=4)
    #     assert isinstance(output, ModelOutput)
    #     assert output.z.shape == (4, 10, model.latent_dim)
    #     assert hasattr(output, "cond_mod_data")
    #     assert output.cond_mod_data.shape == (4, *samples.data["label"].shape)

    #     output = model.encode(samples, N=4, flatten=True)
    #     assert isinstance(output, ModelOutput)
    #     assert output.z.shape == (4 * 10, model.latent_dim)
    #     assert hasattr(output, "cond_mod_data")
    #     assert torch.all(output.cond_mod_data == torch.cat([samples.data["label"]] * 4))

    #     return

    def test_predict(self, model, dataset):

        samples = dataset[:10]

        # Test reconstruction
        output = model.predict(cond_mod="m0", gen_mod="m1", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m1")
        assert output.m1.shape == samples.data["m1"].shape

        output = model.predict(cond_mod="m1", gen_mod="m0", inputs=samples, N=10)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert output.m0.shape == (10, 10, 3, 28, 28)

        output = model.predict(cond_mod=["m0"], gen_mod="m1", inputs=samples)
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m1")
        assert output.m1.shape == samples.data["m1"].shape

        output = model.predict(
            cond_mod=["m0", "m1"], gen_mod="all", inputs=samples
        )
        assert isinstance(output, ModelOutput)
        assert hasattr(output, "m0")
        assert hasattr(output, "m1")

        assert output.m0.shape == samples.data["m0"].shape
        assert output.m1.shape == samples.data["m1"].shape


        
        return

    @fixture(params=[[32, 64, 3, "Adagrad"], [16, 16, 4, "Adam"]])
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
    def trainer(self, trainer_config, model, dataset):

        return BaseTrainer(
            model,
            train_dataset=dataset,
            eval_dataset=dataset,
            training_config=trainer_config,
        )

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
