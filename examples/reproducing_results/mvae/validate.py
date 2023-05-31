from pathlib import Path

import torch
from torch import nn

from multivae.data.datasets.celeba import CelebAttr
from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models import AutoModel

# model = AutoModel.load_from_hf_hub("asenella/reproduce_mvae_mnist_1", allow_pickle=True)

# test_set = BinaryMnistLabels(data_path="../data", split="test", random_binarized=True)

# output_dir = './reproduce_mvae_mnist_1/'
# Path(output_dir).mkdir(exist_ok=True)

# ll_config = LikelihoodsEvaluatorConfig(batch_size=512, K=1000, batch_size_k=500, output_dir=output_dir)

# ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)


# ll_module.eval()


test_set = MMNISTDataset(data_path="~/scratch/data", split="test")

data_path = "/home/asenella/dev/multivae_package/multimodal_vaes/dummy_output_dir/MVAE_training_2023-05-23_23-36-27/final_model"

model = AutoModel.load_from_folder(data_path)


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


device = "cuda" if torch.cuda.is_available() else "cpu"
clfs = load_mmnist_classifiers(device=device)

import json

with open(data_path + "/wandb_info.json", "r") as fp:
    dict_w = json.load(fp)

w_path = dict_w["path"]

coherences = CoherenceEvaluator(
    model, clfs, test_set, data_path, CoherenceEvaluatorConfig(wandb_path=w_path)
).eval()
