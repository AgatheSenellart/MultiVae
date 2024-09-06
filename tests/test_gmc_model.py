from multivae.models.gmc import GMC, GMCConfig
import pytest
from multivae.data.datasets import MultimodalBaseDataset
import torch 
import numpy as np
from multivae.models.nn.default_architectures import BaseDictEncoders, MultipleHeadJointEncoder, Encoder_VAE_MLP, ModelOutput, BaseAEConfig
import os


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
    
    @pytest.fixture
    def model_config(self, n_modalities, input_dims):
        return GMCConfig(
            n_modalities=n_modalities,
            input_dims=input_dims,
            common_dim= 10,
            latent_dim = 5,
            temperature=0.2
        )
    
    
    @pytest.fixture
    def encoders(self, input_dims, model_config):
        """Basic encoders for each modality"""
        
        return BaseDictEncoders(
            input_dims=input_dims,
            latent_dim=model_config.common_dim
        )
        
    
    @pytest.fixture
    def joint_encoder(self, input_dims, model_config):
        
        config = BaseAEConfig(latent_dim=model_config.common_dim)
        
        return MultipleHeadJointEncoder(
            BaseDictEncoders(input_dims=input_dims,latent_dim=100), args=config
        )
        
    @pytest.fixture
    def shared_encoder(self, model_config):
        
        
        return Encoder_VAE_MLP(BaseAEConfig(latent_dim = model_config.latent_dim, input_dim = (model_config.common_dim,)))
    
    @pytest.fixture
    def model(self, model_config,encoders, joint_encoder, shared_encoder):
        
        return GMC(config=model_config, processors=encoders, joint_encoder=joint_encoder, shared_encoder=shared_encoder)
    
    
    def test_forward(self, model, dataset):
        
        output = model(dataset)
        assert isinstance(output, ModelOutput)
        
        loss = output.loss
        
        assert isinstance(loss, torch.Tensor)
        
    
    def test_encode(self, model, dataset):
        
        for cond_mod in ['all','m1']:
            output = model.encode(dataset,cond_mod = cond_mod)
            assert isinstance(output, ModelOutput)
            embedding = output.embedding
            assert isinstance(embedding, torch.Tensor)
            assert embedding.shape == (len(dataset),model.latent_dim)
            
        
        
        

        

        
        
        
        
        
        