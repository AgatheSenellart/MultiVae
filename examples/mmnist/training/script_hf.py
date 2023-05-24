import os
import tempfile

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, FIDEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel
import argparse
import json
from config2 import *
##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="test")

# Get config_files
parser = argparse.ArgumentParser()
parser.add_argument('--param_file',type=str)
args = parser.parse_args()

with open(args.param_file,'r') as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

missing_ratio = ''.join(str(args.missing_ratio).split('.'))
incomplete = 'i' if args.keep_incomplete else 'c'


hf_repo = f"asenella/mmnist_{'JMVAE'}{config_name}_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)
model = model.cuda()
model.device = 'cuda'

import wandb

wandb_run = wandb.init(entity='multimodal_vaes',config=args)

eval_model(model, output_dir=None,test_data=test_set,wandb_path=wandb_run.path)