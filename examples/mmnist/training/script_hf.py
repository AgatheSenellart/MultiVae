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

##############################################################################
train_set = MMNISTDataset(data_path="~/scratch/data", split="train")

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


model = model.cuda()

model.device = "cuda"


import wandb

wandb_run = wandb.init(entity="multimodal_vaes", project='validate_mmnist', config=args.__dict__.update(model.model_config.to_dict()))
output_dir = f'./validate_mmnist/{args.model_name}/incomplete_{incomplete}/missing_ratio_{missing_ratio}/'

# Recompute the cross-coherences and joint coherence from prior
config = CoherenceEvaluatorConfig(batch_size=512, wandb_path=wandb_run.path)

CoherenceEvaluator(
    model=model,
    test_dataset=test_set,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=output_dir,
    eval_config=config,
).eval()

# Compute joint coherence from other samplers

# From Gaussian Mixture Sampler 
from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig
sampler_config = GaussianMixtureSamplerConfig(n_components=10)
sampler = GaussianMixtureSampler(model)
sampler.fit(train_set)

module_eval = CoherenceEvaluator(model,load_mmnist_classifiers(),test_set,eval_config=config,sampler=sampler)
module_eval.joint_coherence()
module_eval.finish()

# Sample some images to visualize
output = sampler.sample(n_samples = 8, batch_size=8)
samples_images = model.decode(output)
tight_array = torch.cat([samples_images[m] for m in modalities])
tight_array = make_grid(tight_array)

ndarr = (
                tight_array.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
recon_image = Image.fromarray(ndarr)
wandb_run.log({f'samples_{sampler.name}' : wandb.Image(recon_image)})
module_eval.finish()


# From IAF sampler
from multivae.samplers import IAFSampler, IAFSamplerConfig
from pythae.trainers import BaseTrainerConfig

training_config = BaseTrainerConfig(per_device_train_batch_size=512, num_epochs=1)
sampler_config = IAFSamplerConfig()
sampler = IAFSampler(model)
sampler.fit(train_set,training_config=training_config)

module_eval = CoherenceEvaluator(model,load_mmnist_classifiers(),test_set, eval_config=config,sampler=sampler)
module_eval.joint_coherence()
module_eval.finish()



# Compute joint likelihood
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig

lik_config = LikelihoodsEvaluatorConfig(
    batch_size=512,
    wandb_path=wandb_run.path,
    num_samples=1000,
    batch_size_k=250,
)

lik_module = LikelihoodsEvaluator(model,
                                  test_set,
                                  output= f'./validate_mmnist/{args.model_name}/incomplete_{incomplete}/missing_ratio_{missing_ratio}/',
                                  eval_config=lik_config,
                                  )
lik_module.eval()