import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.metrics.fids.fids import FIDEvaluator
from multivae.models.auto_model import AutoConfig, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=8)
args = parser.parse_args()


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h


def load_mmnist_classifiers(data_path="../../../data/clf", device="cuda"):
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


##############################################################################

test_set = MMNISTDataset(data_path="~/scratch/data", split="test")

device = "cuda" if torch.cuda.is_available() else "cpu"

clfs = load_mmnist_classifiers("/home/asenella/scratch/data/clf", device=device)
for seed in range(3):
    model = AutoModel.load_from_hf_hub(
        f"asenella/reproducing_mvtcae_seed_{seed}", allow_pickle=True
    )
    # model = AutoModel.load_from_folder(data_path)

    coherences = CoherenceEvaluator(model, clfs, test_set, None).eval()

# fids = FIDEvaluator(model,test_set).mvtcae_reproduce_fids(gen_mod='m0')
