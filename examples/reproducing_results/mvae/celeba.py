import torch
import torch.nn.functional as F
from pythae.models.base.base_model import BaseDecoder, BaseEncoder, ModelOutput
from torch import nn

from multivae.data.datasets.celeba import CelebAttr
from multivae.models import MVAE, MVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

######## Architectures ###########

torch.backends.cudnn.benchmark = True


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)


class ImageEncoder(BaseEncoder):
    """Parametrizes q(z|x).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, n_latents * 2),
        )
        self.n_latents = n_latents
        self.latent_dim = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.classifier(x)
        return ModelOutput(embedding=x[:, :n_latents], log_covariance=x[:, n_latents:])


class ImageDecoder(BaseDecoder):
    """Parametrizes p(x|z).
    This is the standard DCGAN architecture.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(nn.Linear(n_latents, 256 * 5 * 5), Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5)
        z = self.hallucinate(z)
        return ModelOutput(reconstruction=z)  # NOTE: no sigmoid here. See train.py


class AttributeEncoder(BaseEncoder):
    """Parametrizes q(z|y).
    We use a single inference network that encodes
    all 18 features.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(AttributeEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, n_latents * 2),
        )
        self.n_latents = n_latents
        self.latent_dim = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return ModelOutput(embedding=x[:, :n_latents], log_covariance=x[:, n_latents:])


class AttributeDecoder(BaseDecoder):
    """Parametrizes p(y|z).
    We use a single generative network that decodes
    all 18 features.
    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(AttributeDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(512, 18),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.net(z)
        # not a one-hotted prediction: this returns a value
        # for every single index
        return ModelOutput(reconstruction=z)  # NOTE: no sigmoid here. See train.py


#######################################################
#### MODEL

model_config = MVAEConfig(
    n_modalities=2,
    input_dims=dict(image=(3, 64, 64), attributes=(18,)),
    latent_dim=100,
    uses_likelihood_rescaling=True,
    rescale_factors=dict(image=1, attributes=10),
    decoder_dist=dict(image="bernoulli", attributes="bernoulli"),
    warmup=20,
)


model = MVAE(
    model_config,
    encoders=dict(
        image=ImageEncoder(model_config.latent_dim),
        attributes=AttributeEncoder(model_config.latent_dim),
    ),
    decoders=dict(
        image=ImageDecoder(model_config.latent_dim),
        attributes=AttributeDecoder(model_config.latent_dim),
    ),
)


###########################################################
### Training config


training_config = BaseTrainerConfig(
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    learning_rate=1e-4,
    start_keep_best_epoch=model.warmup + 1,
    num_epochs=100,
    steps_predict=1,
    steps_saving=49,
)


train_set = CelebAttr("~/scratch/data", "train", download=True)
eval_set = CelebAttr("~/scratch/data", "valid", download=True)

wandb_cb = WandbCallback()
run_id = "wise-firebrand-14"
wandb_cb.setup(
    training_config,
    model_config,
    project_name="reproduce_mvae",
    run_id=run_id,
    resume="must",
)

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=eval_set,
    training_config=training_config,
    callbacks=callbacks,
    checkpoint="dummy_output_dir/MVAE_training_2023-04-21_16-05-34/checkpoint_epoch_98",
)

trainer.train()

trainer._best_model.push_to_hf_hub("asenella/reproducing_mvae")
