"""In this file, we reproduce the results of the DMVAE model on Mnist-SVHN"""

from architectures import (
    DecoderMNIST,
    DecoderSVHN,
    EncoderMNIST,
    EncoderSVHN,
    load_mnist_svhn_classifiers,
)

from multivae.data.datasets import MnistSvhn
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models import DMVAE, DMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback

# Define the paths for loading the data and saving the model
DATA_PATH = "/home/asenella/data"
SAVE_PATH = "/home/asenella/experiments"
CLASSIFIERS_PATH = "/home/asenella/classifiers"  # Path to trained classifiers for MNIST-SVHN. Trained models can be downloaded from
# https://mybox.inria.fr/d/39d47446eaf9437fbb61/

# Load the dataset
train_set = MnistSvhn(
    data_path=DATA_PATH, split="train", data_multiplication=30, download=True
)
test_set = MnistSvhn(
    data_path=DATA_PATH, split="test", data_multiplication=30, download=True
)

print(f"train : {len(train_set)}, test : {len(test_set)}")

# Model config
model_config = DMVAEConfig(
    n_modalities=2,
    latent_dim=10,
    modalities_specific_dim={"mnist": 1, "svhn": 4},
    rescale_factors={"mnist": 50, "svhn": 1},
    uses_likelihood_rescaling=True,
)


model = DMVAE(
    model_config,
    encoders={
        "mnist": EncoderMNIST(
            256,
            1,
            model_config.latent_dim,
            model_config.modalities_specific_dim["mnist"],
        ),
        "svhn": EncoderSVHN(
            model_config.latent_dim, model_config.modalities_specific_dim["svhn"]
        ),
    },
    decoders={
        "mnist": DecoderMNIST(
            256,
            1,
            model_config.latent_dim,
            model_config.modalities_specific_dim["mnist"],
        ),
        "svhn": DecoderSVHN(
            model_config.latent_dim, model_config.modalities_specific_dim["svhn"]
        ),
    },
)


# Training
training_config = BaseTrainerConfig(
    learning_rate=1e-3,
    per_device_train_batch_size=100,
    per_device_eval_batch_size=100,
    num_epochs=10,
    optimizer_cls="Adam",
    optimizer_params={"amsgrad": True},
    steps_predict=1,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_dmvae")

callbacks = [wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=test_set,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

#### Validate ####

# load the classifiers
classifiers = load_mnist_svhn_classifiers(CLASSIFIERS_PATH)

eval_config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_cb.run.path)
eval_module = CoherenceEvaluator(
    model, classifiers, test_set, trainer.training_dir, eval_config
)

eval_module.eval()
eval_module.log_to_wandb()
eval_module.finish()
