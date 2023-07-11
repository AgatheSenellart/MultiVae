


# Image
from pythae.models.nn.benchmarks.mnist import Encoder_Conv_VAE_MNIST, Decoder_Conv_AE_MNIST

# Sound

import torch
import torch.nn as nn
import torch.nn.functional as F
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput

FRAME_SIZE = 512
CONTEXT_FRAMES = 32
SPECTROGRAM_BINS = FRAME_SIZE//2 + 1


class SoundEncoder(BaseEncoder):
    def __init__(self, output_dim):
        super(SoundEncoder, self).__init__()
        self.latent_dim = output_dim

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
        return ModelOutput(
            embedding = self.fc_mu(h),
            log_covariance = self.fc_logvar(h))


class SoundDecoder(BaseDecoder):
    def __init__(self, input_dim):
        super(SoundDecoder, self).__init__()
        self.latent_dim = input_dim

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
        return ModelOutput(reconstruction = F.sigmoid(out))



class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)
    
    
    
# Trajectory Encoder

import torch
import torch.nn as nn


# Symbol
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
        return ModelOutput(reconstruction = self.out_process(self.network(x)))
    




##### Classifiers #####

class Image_Classifier(nn.Module):
    def __init__(self):
        super(Image_Classifier, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)

        return out
    
    
class Sound_Classifier(nn.Module):
    def __init__(self):
        super(Sound_Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(2048, 128),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 64),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(),
                                nn.Linear(64, 10))




    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out



class Trajectory_Classifier(nn.Module):
    def __init__(self):
        super(Trajectory_Classifier, self).__init__()

        self.network = nn.Sequential(nn.Linear(200, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 128),
                                  nn.BatchNorm1d(128),
                                  nn.LeakyReLU())
        self.out = nn.Linear(128, 10)


    def forward(self, x):
        h = self.network(x)
        return self.out(h)