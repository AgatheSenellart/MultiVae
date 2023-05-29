import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.models.base.base_model import BaseDecoder, BaseEncoder, ModelOutput
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

######## Dataset #########

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")


#### Validation ####
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.fids import FIDEvaluator, FIDEvaluatorConfig


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


def load_mmnist_classifiers(data_path="/home/asenella/scratch/data/clf", device="cuda"):
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


import json

from multivae.models import AutoModel

for seed in range(3):
    path = f"../reproduce_mmvaep/K__{1}/seed__{seed}/final_model"

    model = AutoModel.load_from_folder(path)
    model.device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(path, "wandb_info.json")) as fp:
        wandb_path = json.load(fp)["path"]

    config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_path)

    CoherenceEvaluator(
        model=model,
        test_dataset=test_data,
        classifiers=load_mmnist_classifiers(device=model.device),
        output=path,
        eval_config=config,
    ).eval()

    config = FIDEvaluatorConfig(batch_size=128, wandb_path=wandb_path)

    fid = FIDEvaluator(model, test_data, output=path, eval_config=config).eval()
