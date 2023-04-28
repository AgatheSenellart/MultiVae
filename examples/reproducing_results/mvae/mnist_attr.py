import torch
import torch.nn.functional as F
from torch import nn

from multivae.data.datasets.mnist_labels import BinaryMnistLabels
from multivae.models import MVAE, MVAEConfig
from multivae.models.nn.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
)
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.base_trainer_config import BaseTrainerConfig
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

###############################################################
###### Encoders & Decoders


def labels_to_binary_tensors(labels):
    tensor_labels = torch.zeros((len(labels), 10)).float().to(labels.device)
    for i in range(10):
        tensor_value = torch.zeros(10).float().to(labels.device)
        tensor_value[i] = 1.0
        tensor_labels[labels == i] = tensor_value
    return tensor_labels


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)


class ImageEncoder(BaseEncoder):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        self.latent_dim = n_latents
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        return ModelOutput(embedding=self.fc31(h), log_covariance=self.fc32(h))


class ImageDecoder(BaseDecoder):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.latent_dim = n_latents
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        h = self.sigmoid(self.fc4(h))
        return ModelOutput(reconstruction=h.reshape(-1, 1, 28, 28))


class TextEncoder(BaseEncoder):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.latent_dim = n_latents
        self.fc1 = nn.Embedding(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = labels_to_binary_tensors(x)
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return ModelOutput(embedding=self.fc31(h), log_covariance=self.fc32(h))


class TextDecoder(BaseDecoder):
    """Parametrizes p(y|z).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.latent_dim = n_latents
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.swish = Swish()
        self.softmax = nn.Softmax()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return ModelOutput(reconstruction=self.softmax(self.fc4(h)))


#########################################################
#### Model

model_config = MVAEConfig(
    n_modalities=2,
    latent_dim=64,
    input_dims=dict(images=(1, 28, 28), labels=(1,)),
    decoders_dist=dict(images="bernoulli", labels="categorical"),
    warmup=200,
    uses_likelihood_rescaling=True,
    rescale_factors=dict(images=1, labels=50),
)

encoders = dict(
    images=ImageEncoder(model_config.latent_dim),
    labels=TextEncoder(model_config.latent_dim),
)
decoders = dict(
    images=ImageDecoder(model_config.latent_dim),
    labels=TextDecoder(model_config.latent_dim),
)

model = MVAE(model_config, encoders, decoders)

######################################################
### Dataset

train_set = BinaryMnistLabels(data_path="../../../data", split="train")
test_set = BinaryMnistLabels(data_path="../../../data", split="test")

##############################################################
#### Training


training_config = BaseTrainerConfig(
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_epochs=500,
    start_keep_best_epoch=model_config.warmup,
    steps_predict=5,
    learning_rate=1e-3,
)
wandb_ = WandbCallback()
wandb_.setup(training_config, model_config, project_name="reproduce_mvae_mnist")
callbacks = [wandb_, ProgressBarCallback()]

trainer = BaseTrainer(
    model,
    train_set,
    eval_dataset=test_set,
    training_config=training_config,
    callbacks=callbacks,
    checkpoint=None,
)

trainer.train()

trainer._best_model.push_to_hf_hub("asenella/reproduce_mvae_mnist")
