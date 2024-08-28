import torch
import torch.nn as nn
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput


# Symbol
class TrajectoryEncoder(BaseEncoder):
    def __init__(self, name, input_dim, layer_sizes, output_dim):
        super(TrajectoryEncoder, self).__init__()

        # Variables
        self.name = name
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
        print('Info:' + str(self.name))
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding = self.fc_mu(h), log_covariance = self.fc_logvar(h))


class TrajectoryDecoder(BaseDecoder):
    def __init__(self, name, input_dim, layer_sizes, output_dim):
        super(TrajectoryDecoder, self).__init__()

        # Variables
        self.name = name
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
        print('Info:' + str(self.name))
        print(f'Layers: {dec_layers}')


    def forward(self, x):
        return ModelOutput(reconstruction = self.out_process(self.network(x)))