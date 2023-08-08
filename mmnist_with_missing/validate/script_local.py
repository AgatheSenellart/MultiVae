import os

import numpy as np
import torch
from classifiers import load_mmnist_classifiers
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel

##############################################################################

test_set = MMNISTDataset(data_path="../../../data", split="test")

data_path = "dummy_output_dir/JNF_training_2023-04-01_23-43-05/final_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
clfs = load_mmnist_classifiers(device=device)

model = AutoModel.load_from_folder(data_path)

output = CoherenceEvaluator(model, clfs, test_set, data_path).eval()
# model.push_to_hf_hub('asenella/mmnist'+ model.model_name + '_config2_')
