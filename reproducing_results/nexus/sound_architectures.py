import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from multivae.models.base import BaseDecoder, BaseEncoder,ModelOutput

FRAME_SIZE = 512
CONTEXT_FRAMES = 32
SPECTROGRAM_BINS = FRAME_SIZE//2 + 1


class SoundEncoder_network(nn.Module):
    def __init__(self, output_dim):
        super(SoundEncoder_network, self).__init__()

        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, output_dim)
        self.fc_logvar = nn.Linear(2048, output_dim)


    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class SoundDecoder_network(nn.Module):
    def __init__(self, input_dim):
        super(SoundDecoder_network, self).__init__()

        self.upsampler = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)
        return F.sigmoid(out)



class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor





class SigmaVAE(nn.Module):
    def __init__(self, latent_dim, use_cuda=False):

        super(SigmaVAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda

        # Components
        self.mod_encoder = SoundEncoder_network(output_dim=self.latent_dim)
        self.mod_decoder = SoundDecoder_network(input_dim=self.latent_dim)


    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)


    def forward(self, x):
        mu, logvar = self.mod_encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.mod_decoder(z)

        # Compute log_sigma optimal
        log_sigma = ((x - out) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()

        # log_sigma = self.log_sigma
        out_log_sigma = softclip(log_sigma, -6)

        return out, out_log_sigma, z, mu, logvar
    
    

s_vae = SigmaVAE(128, use_cuda=torch.cuda.is_available())
s_vae.load_state_dict(torch.load('/home/asenella/dev/multivae_package/best_sound_vae_model.pth.tar', 
                                 map_location='cpu')['state_dict'])


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
            
            
class SoundDecoder(BaseDecoder):
    def __init__(self, output_dim):
        super(SoundDecoder, self).__init__()
        self.output_dim = output_dim
        self.network = s_vae.mod_decoder

    def forward(self,x):
        with torch.no_grad():
            recon = self.network(x)
            return ModelOutput(
                reconstruction = recon)