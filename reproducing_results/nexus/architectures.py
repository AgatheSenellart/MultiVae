from multivae.models import Nexus, NexusConfig
from multivae.data.datasets.mhd import MHD
from multivae.models.base import BaseEncoder, BaseAEConfig, BaseDecoder, ModelOutput
from torch import nn 
import torch
from math import prod
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sigma_vae import *

# define the architectures

def convolutional_output_width(input_width, kernel_width, padding, stride):
    # assumes square input/output and kernels
    return int((input_width - kernel_width + 2 * padding) / stride + 1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Bottom encoders
class ImageEncoder(BaseEncoder):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential( 
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish()
        )
        
       
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            Swish(),
            nn.Linear(128,128),
            Swish()
        )
   
        # Output layer of the network
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_logvar = nn.Linear(128, self.latent_dim) 
        
    def forward(self, x):
        x = self.conv_layers(x)
        h = x.view(x.size(0), -1)
        h = self.linear_layers(h)
        return ModelOutput(embedding = self.fc_mu(h), log_covariance = self.fc_logvar(h))  
    
class SoundEncoder(BaseEncoder):
    def __init__(self, output_dim):
        super(SoundEncoder, self).__init__()
        self.latent_dim = output_dim
        self.network = s_vae.mod_encoder

    def forward(self,x):
        with torch.no_grad():
            embedding, log_var = self.network(x)
            return ModelOutput(
                embedding = embedding,
                log_covariance = log_var)

class TrajectoryEncoder(BaseEncoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryEncoder, self).__init__()
        self.latent_dim = output_dim

        # Variables
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        enc_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]
            enc_layers.append(nn.Linear(pre, pos))
            enc_layers.append(nn.BatchNorm1d(pos))
            enc_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        # Output layer of the network
        self.fc_mu = nn.Linear(pre, output_dim)
        self.fc_logvar = nn.Linear(pre, output_dim)

        # Print information
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding = self.fc_mu(h),
                           log_covariance= self.fc_logvar(h))

class LabelEncoder(BaseEncoder):
    
    def __init__(self, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(10,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
        )
        
        self.fc_embedding = nn.Linear(128,latent_dim)
        self.fc_covariance = nn.Linear(128,latent_dim)
    
    def forward(self, x):
        h = self.layers(x)
        return ModelOutput(
            embedding = self.fc_embedding(h),
            log_covariance = self.fc_covariance(h)
        )
    
    
# Bottom decoders

class ImageDecoder(BaseDecoder):
    def __init__(self,input_dim, latent_dim):
        super(ImageDecoder, self).__init__()

        # Variables
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Linear layers =
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 128), Swish(),
            nn.Linear(128, 128), Swish(),
            nn.Linear(128,3136), Swish(),
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                    64, 32, 4, 2, 1, bias=False), Swish(),
            nn.ConvTranspose2d(
                    32, 1, 4, 2, 1, bias=False), nn.Sigmoid(),
        )
        
    def forward(self, z):
        x = self.linear_layers(z)
        x = x.view(-1, 64, 7,7)
        out = self.conv_layers(x)
        return ModelOutput(reconstruction = out)

class SoundDecoder(BaseDecoder):
    def __init__(self, input_dim):
        super(SoundDecoder, self).__init__()
        self.network = s_vae.mod_decoder

    def forward(self, z):
        
        with torch.no_grad():
            recon = self.network(z)
            return ModelOutput(reconstruction =recon)
        
class TrajectoryDecoder(BaseDecoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryDecoder, self).__init__()
        self.latent_dim = input_dim

        # Variables
        self.id = id
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        dec_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]

            # Check for input transformation
            dec_layers.append(nn.Linear(pre, pos))
            dec_layers.append(nn.BatchNorm1d(pos))
            dec_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        dec_layers.append(nn.Linear(pre, output_dim))
        self.network = nn.Sequential(*dec_layers)

        # Output Transformation
        self.out_process = nn.Sigmoid()

        # Print information
        print(f'Layers: {dec_layers}')


    def forward(self, x):
        
        batch_shape = x.shape[:-1]
        
        x = x.reshape(prod(batch_shape),-1)
        
        out = self.network(x)
        
        if len(batch_shape)>1:
            out = out.reshape(*batch_shape,*out.shape[1:])
        
        return ModelOutput(reconstruction = self.out_process(out))
    
class LabelDecoder(BaseDecoder):
    
    def __init__(self, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(latent_dim,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,10), nn.Softmax()
        )

    def forward(self, z):
        return ModelOutput(reconstruction = self.layers(z))
    
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
            nn.Linear(msg_dim,512),nn.BatchNorm1d(512),nn.LeakyReLU()
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