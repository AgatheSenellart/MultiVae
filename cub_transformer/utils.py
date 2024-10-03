from dataset import CUB
from torch import nn
import torch
from pythae.models.base.base_model import BaseEncoder, ModelOutput
import torch.nn.functional as F

max_length = 32

# dataset
train_data = CUB('/home/asenella/scratch/data', split='train',max_lenght=max_length, one_hot=False)
eval_data = CUB('/home/asenella/scratch/data', split='eval',max_lenght=max_length, one_hot=False)

vocab_size = train_data.text_data.vocab_size



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

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