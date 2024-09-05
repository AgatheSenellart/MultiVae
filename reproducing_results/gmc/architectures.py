from multivae.models.base import BaseEncoder, BaseDecoder, BaseJointEncoder, ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F

##### Architectures #####

class MHDCommonEncoder(BaseEncoder):

    def __init__(self, common_dim, latent_dim):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return ModelOutput(embedding = F.normalize(self.feature_extractor(x), dim=-1))


class MHDImageProcessor(BaseEncoder):

    def __init__(self, common_dim):
        super(MHDImageProcessor, self).__init__()
        self.latent_dim = common_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return ModelOutput(embedding = self.projector(h))


class MHDSoundProcessor(BaseEncoder):
    def __init__(self, common_dim):
        super(MHDSoundProcessor, self).__init__()
        self.latent_dim = common_dim

        # Properties
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.projector = nn.Linear(2048, common_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return ModelOutput(embedding = self.projector(h))


class MHDTrajectoryProcessor(BaseEncoder):
    def __init__(self, common_dim):
        super(MHDTrajectoryProcessor, self).__init__()
        self.latent_dim = common_dim
        
        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        # Output layer of the network
        self.projector = nn.Linear(512, common_dim)

    def forward(self, x):
        h = self.trajectory_features(x)
        return ModelOutput(embedding=self.projector(h))


class MHDLabelProcessor(BaseEncoder):

    def __init__(self, common_dim):
        super(MHDLabelProcessor, self).__init__()
        self.latent_dim = common_dim
        self.projector = nn.Linear(10, common_dim)

    def forward(self, x):
        return ModelOutput(embedding = self.projector(x))



class MHDJointProcessor(BaseJointEncoder):
    """
    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, common_dim):
        super(MHDJointProcessor, self).__init__()
        self.latent_dim = common_dim


        # Modality-specific features
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )

        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.projector = nn.Linear(128 * 7 * 7 + 2048 + 512 + 10, common_dim)

    def forward(self, x : dict):
        x_img, x_sound, x_trajectory, x_label = x['image'],x['audio'],x['trajectory'],x['label']

        # Image
        h_img = self.img_features(x_img)
        h_img = h_img.view(h_img.size(0), -1)

        # Sound
        h_sound = self.sound_features(x_sound)
        h_sound = h_sound.view(h_sound.size(0), -1)

        # Trajectory
        h_trajectory = self.trajectory_features(x_trajectory)

        return ModelOutput(embedding = self.projector(torch.cat((h_img, h_sound, h_trajectory, x_label), dim=-1)))



"""


Extra components


"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)