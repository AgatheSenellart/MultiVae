from multivae.models import Nexus, NexusConfig
from multivae.data.datasets.mhd import MHD
from multivae.models.base import BaseEncoder, BaseAEConfig, BaseDecoder, ModelOutput
from torch import nn 
import torch
from math import prod
from torch.utils.data import DataLoader
import torch.nn.functional as F


# define the top and joint architectures



# Top encoders

class TopEncoder(BaseEncoder):
    
    def __init__(self, input_dim, msg_dim):
        super().__init__()
        
        self.layers = nn.Linear(input_dim,msg_dim)
    def forward(self, z):
        return ModelOutput(embedding = self.layers(z), log_covariance = None)


# Joint encoder

class JointEncoder(BaseEncoder):
    
    def __init__(self, msg_dim, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(msg_dim,512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512,512),nn.BatchNorm1d(512),nn.LeakyReLU()
        )
        self.fc_embedding = nn.Linear(512,latent_dim)
        self.fc_covariance = nn.Linear(512,latent_dim)
        
    def forward(self, z):
        h = self.layers(z)
        return ModelOutput(
            embedding = self.fc_embedding(h),
            log_covariance = self.fc_covariance(h)
        )

# Top decoders

class Topdecoder(BaseDecoder):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, input_dim)
            

        )

    def forward(self, z):
        
        return ModelOutput(reconstruction = self.layers(z))