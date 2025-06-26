import torch
import math
from torch import nn
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput

def soft_clamp(x: torch.Tensor, v: int = 10):
    return x.div(v).tanh_().mul(v)


# First encoder layer

class Encoder(BaseEncoder):
    """Double Conv layer"""
    def __init__(self, base_num_features):
        super().__init__()
        
        self.conv_network = torch.nn.Sequential(
            torch.nn.Conv2d(
                        in_channels=1, 
                        out_channels=base_num_features,
                        kernel_size=3,
                        stride=1, 
                        padding=1,
                        bias=True),
            nn.ReLU(),
            nn.Conv2d(base_num_features, base_num_features,3,1,1, bias=True),
            nn.ReLU())
        
    def forward(self, x):
        return ModelOutput(embedding=self.conv_network(x))
        
        
class ContextLayers(nn.Module):
    
    def __init__(self, n_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.network(x)
    

       
class BottomUpBlock(nn.Module):
    """
    downsampling layer + next context convolution layers
    """

    def __init__(self, n_channels_input, n_channels_output):
        super(BottomUpBlock, self).__init__()

        self.downsample = nn.Conv2d(n_channels_input,n_channels_output, kernel_size=3,stride=2,padding=1, bias=True)
        self.conv_layers = ContextLayers(n_channels_output)
        
    def forward(self, x):
        out = self.downsample(x)
        return self.conv_layers(out)
    
    
class LastBottomUp(BaseEncoder):
    """downsampling layers + next context convolution layers + linear to feature map"""
    
    def __init__(self, n_channels_input, n_channels_output, in_features_linear,latent_dim):
        super().__init__()
        self.downsample_and_conv = BottomUpBlock(n_channels_input, n_channels_output)
        self.mu= nn.Linear(in_features_linear, latent_dim)
        self.lv= nn.Linear(in_features_linear,latent_dim)
        
    def forward(self, x):
        out = self.downsample_and_conv(x)
        out = torch.nn.Flatten()(out)
        return ModelOutput(embedding = self.mu(out), log_covariance = self.lv(out))


### Decoder blocks

class TopDown(nn.Module):
    """Upsample + conv layers"""
    def __init__(self, n_input_channels, n_output_channels,output_size):
        super().__init__()
        
        self.upsample = nn.Upsample(output_size)
        self.layers = nn.Sequential(
        nn.Conv2d(n_input_channels, n_input_channels, 3, 1, 1), 
        nn.ReLU(),
        nn.Conv2d(n_input_channels, n_output_channels, 3, 1, 1), 
        nn.ReLU())
        
    def forward(self, x):
        return self.layers(self.upsample(x))
    

class FirstTopDown(nn.Module):
    """Linear, Unflatten and top_down"""
    
    def __init__(self, latent_dim,reshape_size, n_input_channels, n_output_channels, output_size):
        super().__init__()
        self.feature_map_up = nn.Sequential(
            nn.Linear(latent_dim, math.prod(reshape_size)), 
            nn.Unflatten(1, reshape_size)
        )
        self.top_down = TopDown(n_input_channels,n_output_channels, output_size)
        
    def forward(self, x):
        return self.top_down(self.feature_map_up(x))
    
    
class prior_block(BaseEncoder):
    """Convolutions for the prior blocks"""
    def __init__(self, n_channels):
        super().__init__()

        self.mu = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
        self.logvar = nn.Conv2d(n_channels, n_channels, 1, 1, 0)

    def forward(self, x):
        return ModelOutput(
            embedding=soft_clamp(self.mu(x)), log_covariance=soft_clamp(self.logvar(x))
        )
        
        
class posterior_block(BaseEncoder):
    """Convolutions for the posterior blocks."""
    
    def __init__(self, n_channels_before_concat):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                2 * n_channels_before_concat,
                n_channels_before_concat,
                3,
                1,
                1,
                bias=True,
            ),
            nn.ReLU(),
        )

        self.mu = nn.Conv2d(n_channels_before_concat, n_channels_before_concat, 1, 1, 0)
        self.logvar = nn.Conv2d(
            n_channels_before_concat, n_channels_before_concat, 1, 1, 0
        )

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(
            embedding=soft_clamp(self.mu(h)), log_covariance=soft_clamp(self.logvar(h))
        )
        

class Decoder(BaseDecoder):
    """Last decoder: takes as input feature maps as large as the original image"""
    
    def __init__(self, n_channels, nb_of_blocks=3):
        super().__init__()
        
        
        self.layers = [nn.Conv2d(n_channels, n_channels, 3, 1, 1, 1, bias=True), 
            nn.ReLU()]*nb_of_blocks
        
        self.network = nn.Sequential(*self.layers)
        self.last_conv = nn.Conv2d(n_channels, 1, 3, 1, 1, 1)
        
    def forward(self, x):
        return ModelOutput(reconstruction = self.last_conv(self.network(x)))