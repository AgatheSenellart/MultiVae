"""
This is the main file for training the MMVAE +  model on PolyMNIST with missing data.

"""

import argparse

import torch
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import (
    CoherenceEvaluator,
    CoherenceEvaluatorConfig,
    Visualization,
    VisualizationConfig,
)
from multivae.metrics.classifiers.mmnist import load_mmnist_classifiers
from multivae.models import MMVAEPlus, MMVAEPlusConfig
from multivae.models.nn.mmnist import DecoderResnetMMNIST, EncoderResnetMMNIST
from multivae.trainers import BaseTrainerConfig
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.callbacks import WandbCallback

DATA_PATH = "/home/asenella/data"
SAVE_PATH = "/home/asenella/experiments/mmvaePlus_on_partial"

# Parser to define missing ratio and seed
parser = argparse.ArgumentParser()
parser.add_argument("--missing_ratio", type=float, default=0)
parser.add_argument("--keep_incomplete", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# Import data with missing samples
train_data = MMNISTDataset(
    data_path=DATA_PATH,
    split="train",
    missing_ratio=args.missing_ratio,
    keep_incomplete=args.keep_incomplete,
    download=True,
)

test_data = MMNISTDataset(data_path=DATA_PATH, split="test", download=True)

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed)
)

# Define model
modalities = ["m0", "m1", "m2", "m3", "m4"]
model_config = MMVAEPlusConfig(
    latent_dim=32,
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
    decoder_dist_params={k: {"scale": 0.75} for k in modalities},
    K=1,
    prior_and_posterior_dist="laplace_with_softmax",
    learn_shared_prior=False,
    learn_modality_prior=True,
    beta=2.5,
    modalities_specific_dim=32,
    reconstruction_option="joint_prior",
)


encoders = {
    m: EncoderResnetMMNIST(
        private_latent_dim=model_config.modalities_specific_dim,
        shared_latent_dim=model_config.latent_dim,
    )
    for m in modalities
}
decoders = {
    m: DecoderResnetMMNIST(
        model_config.latent_dim + model_config.modalities_specific_dim
    )
    for m in modalities
}

model = MMVAEPlus(model_config, encoders=encoders, decoders=decoders)

# Define training configuration
trainer_config = BaseTrainerConfig(
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    num_epochs=150,
    optimizer_cls="Adam",
    optimizer_params={},
    steps_predict=5,
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 30},
    seed=args.seed,
    output_dir=f"{SAVE_PATH}/keep_incomplete_{args.keep_incomplete}/missing_ratio_{args.missing_ratio}/",
)

##### Set up callbacks: Uncomment the following lines to use wandb
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mmvae_plus_on_partial")
wandb_cb.run.config.update(args.__dict__)

### Train the model
trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=[wandb_cb],
)
trainer.train()

model = trainer._best_model

### Validate the model and compute metrics

# Coherence evaluator
config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_cb.run.path)
mod = CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(DATA_PATH + "/clf", device=model.device),
    output=trainer.training_dir,
    eval_config=config,
)
mod.eval()
mod.finish()

# Visualize some generated samples
vis_config = VisualizationConfig(
    wandb_path=wandb_cb.run.path, n_samples=8, n_data_cond=10
)
vis_module = Visualization(
    model, test_data, eval_config=vis_config, output=trainer.training_dir
)
vis_module.eval()

# And some conditional samples too
for i in range(2, 5):
    subset = modalities[1 : 1 + i]
    vis_module.conditional_samples_subset(subset)

vis_module.finish()
