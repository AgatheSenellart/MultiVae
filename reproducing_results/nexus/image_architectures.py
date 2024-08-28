import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from functools import reduce
import torch.nn.functional as F
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput

KERNEL_SIZE = 4
STRIDE = 2
PADDING = 1

# Image
class ImageEncoder(BaseEncoder):
    def __init__(self, name, input_dim, n_channels, conv_layers, linear_layers, output_dim):
        super(ImageEncoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = output_dim

        # Create Network
        pre = n_channels
        conv_output_side = input_dim
        self.features = []
        for ls in conv_layers:
            pos = ls
            self.features.append(
                nn.Conv2d(pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.features.append(Swish())
            conv_output_side = convolutional_output_width(
                conv_output_side, KERNEL_SIZE, PADDING, STRIDE)
            pre = pos
        self.features = nn.Sequential(*self.features)
        pos_conv_n_channels = conv_layers[-1]

        # Size of the unrolled images at the end of features
        # pre = pos_conv_n_channels * conv_output_side * conv_output_side
        pre = 64 * 7 * 7                                                                        # Fix this
        self.classifier = []
        for ls in linear_layers:
            pos = ls
            self.classifier.append(nn.Linear(pre, pos))
            self.classifier.append(Swish())
            pre = pos

        # Output layer of the network
        self.fc_mu = nn.Linear(pre, output_dim)
        self.fc_logvar = nn.Linear(pre, output_dim)

        # Print information
        print('Info:' + str(self.name))
        print(f'Conv Layers: {self.features}')
        print(f'Linear Layers: {self.classifier}')
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):

        x = self.features(x)
        h = x.view(x.size(0), -1)
        h = self.classifier(h)
        return ModelOutput(embedding = self.fc_mu(h), log_covariance = self.fc_logvar(h))

# Image Decoder
class ImageDecoder(BaseDecoder):
    def __init__(self, name, input_dim, n_channels, conv_layers, linear_layers, output_dim):
        super(ImageDecoder, self).__init__()

        # Variables
        self.name = name
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.conv_layers = conv_layers
        self.linear_layers = linear_layers
        self.output_dim = output_dim

        # Create Network
        self.conv_output_side = 7
        self.pos_conv_n_channels = 64
        pos_conv_size = self.pos_conv_n_channels * (self.conv_output_side ** 2)

        self.upsampler = []
        pre = input_dim
        mod_linear_layers_sizes = list(linear_layers) + [pos_conv_size]

        for ls in mod_linear_layers_sizes:
            pos = ls
            self.upsampler.append(nn.Linear(pre, pos))
            self.upsampler.append(Swish())
            pre = pos
        self.upsampler = nn.Sequential(*self.upsampler)

        self.hallucinate = []
        pre = conv_layers[0]
        for ls in conv_layers[1:]:
            pos = ls
            self.hallucinate.append(
                nn.ConvTranspose2d(
                    pre, pos, KERNEL_SIZE, STRIDE, PADDING, bias=False))
            self.hallucinate.append(Swish())
            pre = pos
        self.hallucinate.append(
            nn.ConvTranspose2d(
                pre, n_channels, KERNEL_SIZE, STRIDE, PADDING, bias=False))
        self.hallucinate = nn.Sequential(*self.hallucinate)


        # Output Transformation
        self.out_process = nn.Sigmoid()

        # Print information
        print('Info:' + str(self.name))
        print(f'Linear Layers: {self.upsampler}')
        print(f'Conv Layers: {self.hallucinate}')

    def forward(self, x):
        x = self.upsampler(x)
        x = x.view(-1, self.pos_conv_n_channels, self.conv_output_side,
                   self.conv_output_side)
        out = self.hallucinate(x)
        return ModelOutput(reconstruction = self.out_process(out))


# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def convolutional_output_width(input_width, kernel_width, padding, stride):
    # assumes square input/output and kernels
    return int((input_width - kernel_width + 2 * padding) / stride + 1)

def sequence_convolutional_output_width(input_width, conv_layers_sizes,
                                        kernel_width, padding, stride):
    # assumes square input/output and kernels
    output_width = input_width
    for ls in conv_layers_sizes:
        output_width = convolutional_output_width(output_width, kernel_width,
                                                  padding, stride)

    return output_width
