"""Script to compute more evaluation metrics on the MMNIST experiments"""


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
from torchvision.utils import make_grid
from PIL import Image
from multivae.metrics import Visualization, VisualizationConfig


##############################################################################
train_set = MMNISTDataset(data_path="~/scratch/data", split="train")
train_data, eval_data = random_split(
    train_set, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")

# Get config_files
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
info['model_name'] = args.model_name
args = argparse.Namespace(**info)

missing_ratio = "".join(str(args.missing_ratio).split("."))
incomplete = "i" if args.keep_incomplete else "c"


hf_repo = f"asenella/mmnist_{args.model_name}{config_name}_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)

if torch.cuda.is_available():   
    model = model.cuda()
    model.device = "cuda"
else :
    model.cpu()
    model.device ='cpu'


import wandb
id = f'{args.model_name}_{incomplete}_{missing_ratio}_{args.seed}'
wandb_run = wandb.init(entity="multimodal_vaes",
                       project='k_means',
                       config=model.model_config.to_dict()
                       )

wandb.config.update(args)

output_dir = f'./validate_mmnist/{args.model_name}/incomplete_{incomplete}/missing_ratio_{missing_ratio}/seed_{args.seed}'

from multivae.metrics import Clustering, ClusteringConfig

c_config = ClusteringConfig(number_of_runs=10,
    num_samples_for_fit=None, 
    wandb_path=wandb_run.path)
c = Clustering(model, test_set, train_data,eval_config=c_config)
c.eval()