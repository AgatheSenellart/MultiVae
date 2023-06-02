import argparse

import torch

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.models import JMVAE, JMVAEConfig
from multivae.models.nn.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
)
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

######################################################
### Encoders & Decoders

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


class EncoderImage(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )
        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        h = torch.flatten(x, start_dim=-3, end_dim=-1)
        h = self.layers(h)
        emb = self.embedding_layer(h)
        log_var = torch.log(self.var_layer(h))

        return ModelOutput(embedding=emb, log_covariance=log_var)


class EncoderLabels(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )
        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        x = labels_to_binary_tensors(x)

        h = self.layers(x)
        emb = self.embedding_layer(h)
        log_var = torch.log(self.var_layer(h))

        return ModelOutput(embedding=emb, log_covariance=log_var)


class JointEncoder(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.head_image = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512), torch.nn.ReLU()
        )

        self.head_labels = torch.nn.Sequential(
            torch.nn.Linear(10, 512), torch.nn.ReLU()
        )

        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * 2, 512), torch.nn.ReLU()
        )

        self.embedding_layer = torch.nn.Linear(512, latent_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, latent_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        images = x["images"].flatten(start_dim=-3, end_dim=-1)
        labels = labels_to_binary_tensors(x["labels"])

        h1 = self.head_image(images)
        h2 = self.head_labels(labels)
        h = torch.cat((h1, h2), dim=-1)
        h = self.shared_layer(h)

        return ModelOutput(
            embedding=self.embedding_layer(h),
            log_covariance=torch.log(self.var_layer(h)),
        )


class ImageDecoder(BaseDecoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.input_dim = (1, 28, 28)
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 28 * 28),
        )

    def forward(self, z):
        h = self.layers(z)
        shape = (*z.shape[:-1],) + self.input_dim
        return ModelOutput(reconstruction=h.reshape(shape))


class LabelsDecoder(BaseDecoder):
    def __init__(self, latent_dim):
        super().__init__()
        self.input_dim = (10,)
        self.latent_dim = latent_dim
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.Softmax(),
        )

    def forward(self, z):
        h = self.layers(z)
        shape = (*z.shape[:-1],) + self.input_dim
        return ModelOutput(reconstruction=h.reshape(shape))


######################################################
### Dataset

train_set = MnistLabels(data_path="../../../data", split="train")
test_set = MnistLabels(data_path="../../../data", split="test")

######################################################
### Model

model_config = JMVAEConfig(
    n_modalities=2,
    latent_dim=64,
    input_dims=dict(images=(1, 28, 28), labels=(1,)),
    decoders_dist=dict(images="bernoulli", labels="categorical"),
    alpha=0.1,
    warmup=200,
    uses_likelihood_rescaling=False,
)


model = JMVAE(
    model_config=model_config,
    encoders=dict(
        images=EncoderImage(model_config.latent_dim),
        labels=EncoderLabels(model_config.latent_dim),
    ),
    decoders=dict(
        images=ImageDecoder(model_config.latent_dim),
        labels=LabelsDecoder(model_config.latent_dim),
    ),
    joint_encoder=JointEncoder(model_config.latent_dim),
)

#########################################################
### Training

training_config = BaseTrainerConfig(
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_epochs=500,
    start_keep_best_epoch=model_config.warmup,
    steps_predict=5,
    seed=args.seed,
    learning_rate=1e-3,
)
wandb_ = WandbCallback()
wandb_.setup(training_config, model_config, project_name="reproduce_jmvae")
callbacks = [wandb_, ProgressBarCallback()]

trainer = BaseTrainer(
    model,
    train_set,
    training_config=training_config,
    callbacks=callbacks,
    checkpoint=None,
)

trainer.train()

trainer._best_model.push_to_hf_hub(f"asenella/reproduce_jmvae_seed_{args.seed}")


############################################################
### Validating

from multivae.data.datasets.mnist_labels import MnistLabels
from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig
from multivae.models import AutoModel

model = trainer._best_model

ll_config = LikelihoodsEvaluatorConfig(
    K=1000, unified_implementation=False, wandb_path=wandb_.run.path
)

ll_module = LikelihoodsEvaluator(model, test_set, eval_config=ll_config)

ll_module.eval()
