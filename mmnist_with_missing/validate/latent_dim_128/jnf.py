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

data_path = "dummy_output_dir/JNF_training_2023-03-17_15-29-58/final_model"

clfs = load_mmnist_classifiers()

model = AutoModel.load_from_folder(data_path)

eval = CoherenceEvaluator(model, clfs, test_set, data_path)

eval.pair_accuracies()
eval.all_one_accuracies()
# eval.joint_nll()
