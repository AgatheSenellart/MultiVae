import os

import numpy as np
import torch
from classifiers import load_mmnist_classifiers
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import tempfile

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, FIDEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel
from huggingface_hub import CommitOperationAdd, HfApi


##############################################################################

test_set = MMNISTDataset(data_path="../../../data/MMNIST", split="test")

hf_repo = "asenella/mmnistMVTCAE_config1_"
model = AutoModel.load_from_hf_hub(hf_repo, allow_pickle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clfs = load_mmnist_classifiers(device=device)

tmp_output = tempfile.mkdtemp()

# output = CoherenceEvaluator(model, clfs, test_set, tmp_output).eval()
output = FIDEvaluator(model,test_set).eval()

# api = HfApi()
# api.upload_file(
#     path_or_fileobj=tmp_output + '/metrics.log',
#     path_in_repo="metrics.log",
#     repo_id=hf_repo,
#     repo_type="model",
# )
