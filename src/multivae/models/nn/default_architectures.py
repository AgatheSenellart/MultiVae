from pythae.models.nn.base_architectures import BaseEncoder
from pythae.models.base import BaseAEConfig
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from copy import deepcopy
from torch import nn
import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput

def BaseDictEncoders(input_dims : dict, latent_dim:int):
    encoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(input_dim = input_dims[mod], latent_dim=latent_dim)
        encoders[mod] = Encoder_VAE_MLP(config)
    return encoders

def BaseDictDecoders(input_dims : dict, latent_dim :int):
    decoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(input_dim = input_dims[mod], latent_dim=latent_dim)
        decoders[mod] = Decoder_AE_MLP(config)
    return decoders

class MultipleHeadJointEncoder(BaseEncoder):
    """
    A default instance of joint encoder created from copying the architectures for the unimodal encoders,
    concatenating their outputs and passing them through a unifying Multi-Layer-Perceptron.

        Args:
            dict_encoders (dict): Contains an instance of BaseEncoder for each modality (key).
            args (dict): config dictionary. Contains the latent dim.
            hidden_dim (int) : Default to 512.
            n_hidden_layers (int) : Default to 2.
        """
    def __init__(self, dict_encoders :nn.ModuleDict, args:dict, hidden_dim=512, n_hidden_layers=2, **kwargs):

        super().__init__()
        
        # Duplicate all the unimodal encoders with identical instances. 
        self.encoders = nn.ModuleDict()
        self.joint_input_dim = 0
        for modality in dict_encoders:
            self.encoders[modality] = deepcopy(dict_encoders[modality])
            self.joint_input_dim += self.encoders[modality].latent_dim
        
        modules = [nn.Sequential(nn.Linear(self.joint_input_dim, hidden_dim), nn.ReLU(True))]
        for i in range(n_hidden_layers-1):
            modules.extend([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))])

        self.enc = nn.Sequential(*modules)
        self.fc1 = nn.Linear(hidden_dim, args.latent_dim)
        self.fc2 = nn.Linear(hidden_dim, args.latent_dim)
        
        self.latent_dim = args.latent_dim
        
    
    def forward(self, x : dict):
        """ 
        Implements the encoding of the data contained in x.

        Args:
            x (dict): Contains a tensor for each modality (key).
        """
        
        assert(np.all(x.keys()==self.encoders.keys()))
        
        modalities_outputs = []
        for mod in self.encoders:
            modalities_outputs.append(self.encoders[mod](x[mod])['embedding'])
        
        # Stack the modalities outputs
        concatened_outputs = torch.cat(modalities_outputs,dim=1)
        h=self.enc(concatened_outputs)
        embedding = self.fc1(h)
        log_covariance = self.fc2(h)
        output = ModelOutput(embedding=embedding, log_covariance=log_covariance)
        
        return output
        
            
        
        


