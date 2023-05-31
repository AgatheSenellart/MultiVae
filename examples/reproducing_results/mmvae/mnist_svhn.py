from math import prod

import torch
from pythae.models.base.base_model import BaseDecoder, BaseEncoder, ModelOutput
from torch import nn

from multivae.data.datasets import MnistSvhn
from multivae.metrics.likelihoods.likelihoods import LikelihoodsEvaluator
from multivae.metrics.likelihoods.likelihoods_config import LikelihoodsEvaluatorConfig
from multivae.models import MMVAE, MMVAEConfig
from multivae.models.base import BaseMultiVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

############ Define the architectures ##############


class EncoderMNIST(BaseEncoder):
    def __init__(self, num_hidden_layers, config: BaseMultiVAEConfig):
        super().__init__()
        # Constants
        self.latent_dim = config.latent_dim
        dataSize = torch.Size([1, 28, 28])
        data_dim = int(prod(dataSize))
        self.hidden_dim = 400
        modules = []
        modules.append(
            nn.Sequential(nn.Linear(data_dim, self.hidden_dim), nn.ReLU(True))
        )
        modules.extend(
            [self.extra_hidden_layer() for _ in range(num_hidden_layers - 1)]
        )
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(self.hidden_dim, config.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, config.latent_dim)

    def extra_hidden_layer(self):
        return nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))

    def forward(self, x):
        h = self.enc(x.view(*x.size()[:-3], -1))  # flatten data
        return ModelOutput(embedding=self.fc21(h), log_covariance=self.fc22(h))


class DecoderMNIST(BaseDecoder):
    """Generate an MNIST image given a sample from the latent space."""

    def __init__(self, num_hidden_layers, config: BaseMultiVAEConfig):
        super().__init__()
        modules = []
        self.hidden_dim = 400
        self.dataSize = torch.Size([1, 28, 28])
        data_dim = int(prod(self.dataSize))
        modules.append(
            nn.Sequential(nn.Linear(config.latent_dim, self.hidden_dim), nn.ReLU(True))
        )
        modules.extend(
            [self.extra_hidden_layer() for _ in range(num_hidden_layers - 1)]
        )
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, data_dim)

    def extra_hidden_layer(self):
        return nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))

    def forward(self, z):
        p = self.fc3(self.dec(z))
        d = torch.sigmoid(p.view(*z.size()[:-1], *self.dataSize))  # reshape data
        d = d.clamp(1e-6, 1 - 1.0e-6)

        return ModelOutput(reconstruction=d)


# Classes
class EncoderSVHN(BaseEncoder):
    def __init__(self, config: BaseMultiVAEConfig):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.latent_dim = config.latent_dim

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, config.latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 4, config.latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        return ModelOutput(
            embedding=self.c1(e).squeeze(), log_covariance=self.c2(e).squeeze()
        )


class DecoderSVHN(BaseDecoder):
    """Generate a SVHN image given a sample from the latent space."""

    def __init__(self, config: BaseMultiVAEConfig):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 32 x 32
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        # consider also predicting the length scale
        return ModelOutput(reconstruction=out)


###################################################################################
########### Load the dataset, configure model and training ########################

# Dataset
train_set = MnistSvhn(split="train", data_multiplication=30)
test_set = MnistSvhn(split="test", data_multiplication=30)


print(f"train : {len(train_set)}, test : {len(test_set)}")
# Model config
model_config = MMVAEConfig(
    n_modalities=2,
    latent_dim=20,
    input_dims={"mnist": (1, 28, 28), "svhn": (3, 32, 32)},
    uses_likelihood_rescaling=True,
    decoders_dist={"mnist": "laplace", "svhn": "laplace"},
    decoder_dist_params={"mnist": {"scale": 0.75}, "svhn": {"scale": 0.75}},
    K=30,
    learn_prior=True,
    prior_and_posterior_dist="laplace_with_softmax",
)


model = MMVAE(
    model_config,
    encoders={
        "mnist": EncoderMNIST(num_hidden_layers=1, config=model_config),
        "svhn": EncoderSVHN(model_config),
    },
    decoders={
        "mnist": DecoderMNIST(num_hidden_layers=1, config=model_config),
        "svhn": DecoderSVHN(config=model_config),
    },
)


# Training

training_config = BaseTrainerConfig(
    learning_rate=1e-3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_epochs=30,
    start_keep_best_epoch=30,  # save the model at each iteration without regards to the loss
    optimizer_cls="Adam",
    optimizer_params={"amsgrad": True},
    steps_predict=1,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

trainer._best_model.push_to_hf_hub("asenella/reproducing_mmvae_5")

#### Validate ####

lik_config = LikelihoodsEvaluatorConfig(
    batch_size=12,
    batch_size_k=1000,
    unified_implementation=False,
    num_samples=1000,
    wandb_path=wandb_cb.run.path,
)
output = LikelihoodsEvaluator(
    model, test_set, trainer.training_dir, eval_config=lik_config
).eval()
