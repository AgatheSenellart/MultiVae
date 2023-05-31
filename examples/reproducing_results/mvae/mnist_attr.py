import argparse

import torch
import torch.nn.functional as F
from torch import nn

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics.likelihoods.likelihoods import LikelihoodsEvaluator
from multivae.metrics.likelihoods.likelihoods_config import LikelihoodsEvaluatorConfig
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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=8)
args = parser.parse_args()


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
        return x * torch.sigmoid(x)


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

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        h = self.fc4(h)  # no sigmoid, we provide logits for stability
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return ModelOutput(reconstruction=self.fc4(h))  # no softmax here


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
    use_subsampling=True,
    k=0,
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

train_set = MnistLabels(
    data_path="../data", split="train", random_binarized=True, download=True
)
test_set = MnistLabels(
    data_path="../data", split="test", random_binarized=True, download=True
)

##############################################################
#### Training


training_config = BaseTrainerConfig(
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_epochs=500,
    start_keep_best_epoch=model_config.warmup + 1,
    steps_predict=5,
    learning_rate=1e-3,
    seed=args.seed,
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

trainer._best_model.push_to_hf_hub(f"asenella/reproduce_mvae_mnist_{args.seed}")


###############################################################################
###### Validate #############

ll_config = LikelihoodsEvaluatorConfig(
    batch_size=512, K=1000, batch_size_k=500, wandb_path=wandb_.run.path
)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
