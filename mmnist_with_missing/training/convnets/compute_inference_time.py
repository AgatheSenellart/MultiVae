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
import wandb

import time
##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")
small_batch = next(iter(DataLoader(test_set,10)))

# Get config_files

for model_name in ['JMVAE', 'JNF', 'JNFDcca', 'MMVAE', 'MoPoE', 'MVTCAE']:

    missing_ratio = "0"
    incomplete = "c"
    seed = 0
    




    hf_repo = f"asenella/mmnist_{model_name}{config_name}_seed_{seed}_ratio_{missing_ratio}_{incomplete}"
    model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)
    
    # wandb configuration
    id = f'{model_name}_{incomplete}_{missing_ratio}_{seed}'
    wandb_run = wandb.init(entity="multimodal_vaes",
                        project='inference_time_mmnist',
                        config=model.model_config.to_dict()
                        )


    if torch.cuda.is_available():   
        model = model.cuda()
        model.device = "cuda"
    else :
        model.cpu()
        model.device ='cpu'



    # Compute cross-modal inference time depending on the number of input modalities
    
    for i in range(1,6):
        cond_mod = modalities[:i]
        t1 = time.time()
        model.predict(small_batch, cond_mod=cond_mod)
        t2 = time.time()
        
        wandb.log(
            {
                'inference_time_with_{i}_modalities' : t2-t1
            }
        )
    
    wandb_run.finish()