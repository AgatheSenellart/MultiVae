from dataset import CUB
from torch import nn
import torch
from pythae.models.base.base_model import BaseEncoder, ModelOutput
import torch.nn.functional as F
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder, Encoder_VAE_MLP, BaseAEConfig
from multivae.trainers import MultistageTrainer, MultistageTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from architectures_image import *
from multivae.models.nn.cub import CubTextEncoder, CubTextDecoderMLP
from pythae.models.base import BaseAEConfig
import argparse
import json
from torch.utils.data import random_split

max_length = 16

# data_path = '/home/asenella/scratch/data'
# save_path = '/home/asenella/experiments/CUB_transformer'

data_path = '/home/agathe/data'
save_path = '/home/agathe/experiments/CUB_transformer'


# dataset
train_data = CUB(data_path, split='train',max_lenght=max_length, one_hot=False)
eval_data = CUB(data_path, split='eval',max_lenght=max_length, one_hot=False)

vocab_size = eval_data.text_data.vocab_size



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
    
class CUBCommonEncoder(BaseEncoder):

    def __init__(self, common_dim, latent_dim):
        super(CUBCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor =  nn.Linear(common_dim, latent_dim)
        

    def forward(self, x):
        return ModelOutput(embedding = F.normalize(self.feature_extractor(x), dim=-1))