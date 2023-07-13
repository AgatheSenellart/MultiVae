from architectures import *
from multivae.models.base import BaseAEConfig
import argparse
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
import json

wandb_project = 'MNIST_SVHN'
config_name = 'ms_config_1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
project_path = './ms_experiments'

base_config = dict(
    n_modalities=2,
    latent_dim=20,
    input_dims=dict(mnist = (1,28,28),
                    svhn = (3,32,32)),
    decoders_dist=dict(mnist = 'laplace', svhn ='laplace'),
    decoder_dist_params=dict(mnist = dict(scale=0.75), svhn=dict(scale=0.75))

)


base_trainer_config = dict(
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_epochs=200,
    learning_rate = 1e-3,
    steps_predict=5

)

from multivae.data.datasets.mnist_svhn import MnistSvhn
from torch.utils.data import random_split
import os

train_set = MnistSvhn('/home/asenella/scratch/data/', split='train', download=True)
test_set = MnistSvhn('/home/asenella/scratch/data/',split='test', download=True)


classifiers_path = '/home/asenella/scratch/classifiers_mnist_svhn'

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
    return {"mnist": c1.to(device).eval(), "svhn": c2.to(device).eval()}

classifiers = load_mnist_svhn_classifiers(classifiers_path, device)


# Architectures 

encoders = dict(
    mnist = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=base_config['latent_dim'],input_dim=(1,28,28))),
    svhn = EncoderSVHN(config=BaseAEConfig(latent_dim=base_config['latent_dim'], input_dim=(3,32,32)))
)

decoders = dict(
    mnist = DecoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=base_config['latent_dim'],input_dim=(1,28,28))),
    svhn = DecoderSVHN(config=BaseAEConfig(latent_dim=base_config['latent_dim'], input_dim=(3,32,32)))
)
    

from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, FIDEvaluator, FIDEvaluatorConfig

def eval(path,model, classifiers, wandb_path):
    
    coherence_config = CoherenceEvaluatorConfig(128, wandb_path=wandb_path)
    CoherenceEvaluator(
        model=model,
        classifiers=classifiers,
        test_dataset=test_set,
        output=path,
        eval_config=coherence_config
        ).eval()
    
    
    config = FIDEvaluatorConfig(batch_size=128, wandb_path=wandb_path)

    FIDEvaluator(
        model, test_set, output=path, eval_config=config
    ).compute_all_conditional_fids(gen_mod="svhn")
    
    