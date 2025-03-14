"""In this file, we reproduce the results of the CMVAE model on the PolyMNIST dataset."""

import torch
from architectures import Dec, Enc, load_mmnist_classifiers
from torch import nn

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models.cmvae import CMVAE, CMVAEConfig
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

###### Set the paths for loading and saving ######
DATA_PATH = "/home/asenella/data"
SAVE_PATH = "/home/asenella/experiments"

###### Define model configuration ########
modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = CMVAEConfig(
    n_modalities=5,
    K=1,
    decoders_dist={m: "laplace" for m in modalities},
    decoder_dist_params={m: dict(scale=0.75) for m in modalities},
    prior_and_posterior_dist="laplace_with_softmax",
    beta=2.5,
    modalities_specific_dim=32,
    latent_dim=32,
    input_dims={m: (3, 28, 28) for m in modalities},
    learn_modality_prior=True,
    number_of_clusters=40,
    loss="iwae_looser",
)

encoders = {
    m: Enc(model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
    for m in modalities
}
decoders = {
    m: Dec(model_config.latent_dim + model_config.modalities_specific_dim)
    for m in modalities
}

model = CMVAE(model_config, encoders, decoders)

######## Load the dataset #########
train_data = MMNISTDataset(data_path=DATA_PATH, split="train")
test_data = MMNISTDataset(data_path=DATA_PATH, split="test")

########## Training #######


training_config = BaseTrainerConfig(
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_epochs=50 if model_config.K == 10 else 250,
    learning_rate=1e-3,
    output_dir=f"{SAVE_PATH}/reproduce_cmvae/K__{model_config.K}",
    steps_predict=5,
    optimizer_cls="Adam",
    optimizer_params=dict(amsgrad=True),
    seed=0,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_cmvae")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=None,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()


######### Validate the model #############

# Compute all cross modal coherences
config = CoherenceEvaluatorConfig(batch_size=512, wandb_path=wandb_cb.run.path)

CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(
        data_path=DATA_PATH + "/clf", device=model.device
    ),
    output=trainer.training_dir,
    eval_config=config,
).eval()
