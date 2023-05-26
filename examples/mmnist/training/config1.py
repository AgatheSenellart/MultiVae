"""
Store in this file all the shared variables for the benchmark on mmnist.
"""

import torch
from pythae.models.base.base_config import BaseAEConfig
from torch import nn
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models import BaseMultiVAEConfig
from multivae.models.nn.mmnist import (
    Decoder_ResNet_AE_MMNIST,
    Encoder_ResNet_VAE_MMNIST,
)
from multivae.trainers import BaseTrainerConfig
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MMNISTDataset(data_path="../../../data", split="train")
test_data = MMNISTDataset(data_path="../../../data", split="test")
train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)


modalities = ["m0", "m1", "m2", "m3", "m4"]

base_config = dict(
    n_modalities=len(modalities),
    latent_dim=128,
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
)
encoder_class = Encoder_ResNet_VAE_MMNIST
encoders = {
    k: Encoder_ResNet_VAE_MMNIST(
        BaseAEConfig(latent_dim=base_config["latent_dim"], input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: Decoder_ResNet_AE_MMNIST(
        BaseAEConfig(latent_dim=base_config["latent_dim"], input_dim=(3, 28, 28))
    )
    for k in modalities
}


base_training_config = dict(
    learning_rate=1e-3,
    per_device_train_batch_size=256,
    num_epochs=400,
    optimizer_cls="Adam",
    optimizer_params={},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 7},
    steps_predict=1,
)

wandb_project = "compare_on_mmnist"
config_name = "_config1_"


#######################################
## Define parameters for the evaluation


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
