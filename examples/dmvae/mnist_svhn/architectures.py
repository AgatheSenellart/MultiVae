"""In this file, we define all the architectures that we need for training the DMVAE model."""

import torch
import torch.nn.functional as F

from torch import nn
from math import prod
from multivae.models.nn.base_architectures import BaseMultilatentEncoder, BaseDecoder, ModelOutput

############ Define the architectures ##############


class EncoderMNIST(BaseMultilatentEncoder):

    """Encoder with shared and private latent spaces."""
    def __init__(self, hidden_dim, num_hidden_layers, latent_dim, style_dim):
        super().__init__()
        # Constants
        self.latent_dim = latent_dim
        dataSize = torch.Size([1, 28, 28])
        data_dim = int(prod(dataSize))
        self.hidden_dim = hidden_dim
        modules = []
        modules.append(
            nn.Sequential(nn.Linear(data_dim, self.hidden_dim), nn.ReLU(True))
        )
        modules.extend(
            [self.extra_hidden_layer() for _ in range(num_hidden_layers - 1)]
        )
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(self.hidden_dim, latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, latent_dim)
        
        self.fc21_style = nn.Linear(self.hidden_dim, style_dim)
        self.fc22_style = nn.Linear(self.hidden_dim, style_dim)

    def extra_hidden_layer(self):
        return nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))

    def forward(self, x):
        h = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        return ModelOutput(embedding=self.fc21(h), log_covariance=self.fc22(h),
                           style_embedding=self.fc21_style(h), style_log_covariance=self.fc22_style(h))


class DecoderMNIST(BaseDecoder):
    """Generate an MNIST image given a sample from the latent space."""

    def __init__(self, hidden_dim,num_hidden_layers, latent_dim, style_dim):
        super().__init__()
        modules = []
        self.hidden_dim = hidden_dim
        self.dataSize = torch.Size([1, 28, 28])
        data_dim = int(prod(self.dataSize))
        modules.append(
            nn.Sequential(nn.Linear(latent_dim+style_dim, self.hidden_dim), nn.ReLU(True))
        )
        modules.extend(
            [self.extra_hidden_layer() for _ in range(num_hidden_layers - 1)]
        )
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, data_dim)

    def extra_hidden_layer(self):
        return nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *self.dataSize))  # reshape data
        d = d.clamp(1e-6, 1 - 1.0e-6)

        return ModelOutput(reconstruction=d)


# Classes
class EncoderSVHN(BaseMultilatentEncoder):
    def __init__(self, latent_dim, style_dim):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.latent_dim = latent_dim

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1))
        
        self.c1 = nn.Linear(512, latent_dim)
        self.c2 = nn.Linear(512, latent_dim)
        
        self.c1_style = nn.Linear(512, style_dim)
        self.c2_style = nn.Linear(512, style_dim)


    def forward(self, x):
        e = self.enc_hidden(x)
        e = e.view(e.shape[0], -1)
        e = self.fc(e)
        return ModelOutput(
            embedding=self.c1(e), log_covariance=self.c2(e),
            style_embedding=self.c1_style(e), style_log_covariance = self.c2_style(e)
        )


class DecoderSVHN(BaseDecoder):
    """Generate a SVHN image given a sample from the latent space."""

    def __init__(self, latent_dim, style_dim):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.dec_hidden = nn.Sequential(
            nn.Linear(latent_dim+style_dim, 256 * 2 * 2),
            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        hiddens = self.dec_hidden(z)
        hiddens = hiddens.view(-1, 256, 2, 2)
        images_mean = self.dec_image(hiddens)
        return ModelOutput(reconstruction=images_mean)

#########################################################################
############## Classifiers for validation ###############################


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_mnist_svhn_classifiers(data_path, device="cuda"):
    c1 = MNIST_Classifier()
    c1.load_state_dict(torch.load(f"{data_path}/mnist_model.pt", map_location=device))
    c2 = SVHN_Classifier()
    c2.load_state_dict(torch.load(f"{data_path}/svhn_model.pt", map_location=device))
    return {"mnist": c1.to(device).eval(), "svhn": c2.to(device).eval()}

