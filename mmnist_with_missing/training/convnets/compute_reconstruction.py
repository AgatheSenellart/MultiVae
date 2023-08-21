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
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
import wandb

import time
##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")
small_batch = next(iter(DataLoader(test_set,10)))

# Get config_files

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
                


# wandb configuration
wandb_run = wandb.init(entity="multimodal_vaes",
                    project='reconstruction_mmnist',
                    config=model.model_config.to_dict()
                    )
wandb_run.config.update(
    dict(missing_ratio = missing_ratio,
            incomplete = incomplete,
            seed = args.seed)
)

if torch.cuda.is_available():   
    model = model.cuda()
    model.device = "cuda"
else :
    model.cpu()
    model.device ='cpu'

if args.seed == 0:
    
    vis_config = VisualizationConfig(
        batch_size=32,
        wandb_path=wandb_run.path,
        n_samples=1,
        n_data_cond=8
    )
    
    vis_module = Visualization(model,
                test_dataset=test_set,
                output=f'/home/asenella/scratch/reconstruction_mmnist/{model.model_name}/{incomplete}/{missing_ratio}',
                eval_config=vis_config
                )
    
    vis_module.conditional_samples_subset(subset=list(model.encoders.keys()))
    vis_module.finish()
    del vis_module, vis_config






recon_config = ReconstructionConfig(
    batch_size=32,
    wandb_path=wandb_run.path,
    metric='MSE'
)

recon_module = Reconstruction(model,
            test_dataset = test_set,
            output=f'/home/asenella/scratch/reconstruction_mmnist/{model.model_name}/{incomplete}/{missing_ratio}',
            eval_config=recon_config
            )

recon_module.eval()
recon_module.finish()

recon_config = ReconstructionConfig(
    batch_size=32,
    wandb_path=wandb_run.path,
    metric='SSIM'
)

recon_module = Reconstruction(model,
            test_dataset = test_set,
            output=f'/home/asenella/scratch/reconstruction_mmnist/{model.model_name}/{incomplete}/{missing_ratio}',
            eval_config=recon_config
            )
recon_module.eval()
recon_module.finish()



