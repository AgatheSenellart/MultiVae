import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel
from classifiers import load_mmnist_classifiers


##############################################################################

test_set = MMNISTDataset(data_path="../../../data/MMNIST", split="test")

data_path = "dummy_output_dir/MMVAE_training_2023-03-16_09-13-10/final_model"

clfs = load_mmnist_classifiers()

model = AutoModel.load_from_folder(data_path)

eval = CoherenceEvaluator(model, clfs, test_set, data_path)

eval.pair_accuracies()
eval.all_one_accuracies()
# eval.joint_nll()
