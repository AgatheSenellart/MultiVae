import argparse
import json
import os
import tempfile

import numpy as np
import torch
from config2 import *
from huggingface_hub import CommitOperationAdd, HfApi
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, FIDEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel
from multivae.models.base.base_model import BaseEncoder, ModelOutput

##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")

# Get config_files
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

missing_ratio = "".join(str(args.missing_ratio).split("."))
incomplete = "i" if args.keep_incomplete else "c"

class EncoderConvMMNIST_adapted(BaseEncoder):
    """
    Adapt so that it works with DCCA
    """

    def __init__(self, model_config: BaseAEConfig):
        super(EncoderConvMMNIST_adapted, self).__init__()
        self.latent_dim = model_config.latent_dim
        self.shared_encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=2, padding=1, bias=True
            ),  # -> (128, 4, 4)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Conv2d(128, self.latent_dim, 4, 2, 0)
        self.class_logvar = nn.Conv2d(128, self.latent_dim, 4, 2, 0)

    def forward(self, x):
        h = self.shared_encoder(x)
        return ModelOutput(
            embedding=self.class_mu(h).squeeze(),
            log_covariance=self.class_logvar(h).squeeze(),
        )

hf_repo = f"asenella/mmnist_{'JMVAE'}{config_name}_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)


model = model.cuda()

model.device = "cuda"


import wandb

wandb_run = wandb.init(entity="multimodal_vaes", project='validate_jmvae_mmnist', config=args)

eval_model(model, output_dir=None, test_data=test_set, wandb_path=wandb_run.path)
