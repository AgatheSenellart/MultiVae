import torch
from pythae.models.base.base_config import BaseAEConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.utils import save_all_images
from multivae.data.utils import set_inputs_to_device
from multivae.metrics.coherences import CoherenceEvaluator
from multivae.models.base import BaseMultiVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MnistSvhn(
    data_path="scratch/asenella/data/",
    split="train",
    data_multiplication=5,
    download=True,
)
test_data = MnistSvhn(
    data_path="scratch/asenella/data/",
    split="test",
    data_multiplication=5,
    download=True,
)
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

base_model_config = dict(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    uses_likelihood_rescaling=True,
    decoders_dist=dict(mnist="laplace", svhn="laplace"),
)

encoders = dict(
    mnist=Encoder_VAE_MLP(
        BaseAEConfig(latent_dim=base_model_config["latent_dim"], input_dim=(1, 28, 28))
    ),
    svhn=Encoder_VAE_SVHN(
        BaseAEConfig(latent_dim=base_model_config["latent_dim"], input_dim=(3, 32, 32))
    ),
)

decoders = dict(
    mnist=Decoder_AE_MLP(
        BaseAEConfig(latent_dim=base_model_config["latent_dim"], input_dim=(1, 28, 28))
    ),
    svhn=Decoder_VAE_SVHN(
        BaseAEConfig(latent_dim=base_model_config["latent_dim"], input_dim=(3, 32, 32))
    ),
)

base_training_config = dict(
    learning_rate=1e-3,
    per_device_train_batch_size=256,
    num_epochs=200,
    optimizer_cls="Adam",
    optimizer_params={},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 10},
    steps_predict=1,
)

wandb_project = "compare_on_mnist_svhn"
config_name = "_config1_"


#######################################################################
### Classifiers for evaluation


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_mnist_svhn_classifiers(data_path, device="cuda"):
    c1 = MNIST_Classifier()
    c1.load_state_dict(torch.load(f"{data_path}/mnist.pt", map_location=device))
    c2 = SVHN_Classifier()
    c2.load_state_dict(torch.load(f"{data_path}/svhn.pt", map_location=device))
    return {"mnist": c1.to(device), "svhn": c2.to(device)}


data_path_classifiers = "../../classifiers"
