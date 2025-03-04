"""
Store in this file all the shared variables for the benchmark on mmnist.
"""

import argparse

import torch
from torch.utils.data import random_split

from multivae.data.datasets import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.classifiers.mmnist import load_mmnist_classifiers
from multivae.metrics.fids.fids import FIDEvaluator
from multivae.metrics.fids.fids_config import FIDEvaluatorConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from multivae.trainers import BaseTrainer, BaseTrainerConfig

# imports useful for all scripts
from multivae.trainers.base.callbacks import WandbCallback

# Set your paths
DATA_PATH = "/home/asenella/data"
SAVE_PATH = "/home/asenella/experiments/benchmark_on_partial_mmnist"
FID_PATH = DATA_PATH + "/pt_inception-2015-12-05-6726825d.pth"
CLASSIFIER_PATH = DATA_PATH + "/clf"

WANDB_PROJECT = "benchmark_on_partial_polymnist"


# Shared configuration variables
modalities = ["m0", "m1", "m2", "m3", "m4"]
base_config = dict(
    n_modalities=len(modalities),
    latent_dim=512,
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
    decoder_dist_params={k: {"scale": 0.75} for k in modalities},
)

# Shared Encoders, Decoders
encoders = {
    k: EncoderConvMMNIST_adapted(
        BaseAEConfig(
            latent_dim=base_config["latent_dim"], style_dim=0, input_dim=(3, 28, 28)
        )
    )
    for k in modalities
}

decoders = {
    k: DecoderConvMMNIST(
        BaseAEConfig(
            latent_dim=base_config["latent_dim"], style_dim=0, input_dim=(3, 28, 28)
        )
    )
    for k in modalities
}

# Shared training config
base_training_config = dict(
    learning_rate=1e-3,
    per_device_train_batch_size=256,
    num_epochs=800,
    optimizer_cls="Adam",
    optimizer_params={},
    steps_predict=5,
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 30},
)


# Define everything we want to do in the evaluation in one tight function
def eval_model(model, output_dir, test_data, wandb_path):
    """
    In this function, define all the evaluation metrics
    you want to use
    """

    # Coherence
    config = CoherenceEvaluatorConfig(batch_size=512, wandb_path=wandb_path)
    coherence_module = CoherenceEvaluator(
        model=model,
        test_dataset=test_data,
        classifiers=load_mmnist_classifiers(device=model.device),
        output=output_dir,
        eval_config=config,
    )
    coherence_module.eval()
    coherence_module.finish()

    # FID
    config = FIDEvaluatorConfig(
        batch_size=512, wandb_path=wandb_path, inception_weights_path=FID_PATH
    )

    fid_module = FIDEvaluator(model, test_data, output=output_dir, eval_config=config)
    fid_module.compute_all_conditional_fids(gen_mod="m0")
    fid_module.finish()


# utils


def argument_parser():
    """argument parser for the experiment parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep_incomplete", action="store_true")
    parser.add_argument("--missing_ratio", type=float)
    return parser


def get_datasets(args=None):
    """Return the dataset with the right missing_ratio and the keep_incomplete corresponding to args"""
    if args is None:
        args = argparse.Namespace(missing_ratio=0, keep_incomplete=True)

    train_data = MMNISTDataset(
        data_path=DATA_PATH,
        split="train",
        missing_ratio=args.missing_ratio,
        keep_incomplete=args.keep_incomplete,
    )

    test_data = MMNISTDataset(data_path=DATA_PATH, split="test")

    train_data, eval_data = random_split(
        train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
    )

    return train_data, eval_data, test_data


def model_save_path(model, args):
    """path manufacturer for saving models in a structured way"""
    return f"{SAVE_PATH}/{model.model_name}/keep_incomplete_{args.keep_incomplete}/missing_ratio_{args.missing_ratio}/seed_{args.seed}"


def get_hf_path_from_arguments(args):
    """Return the hf path corresponding to the arguments in args."""
    missing_ratio = "".join(str(args.missing_ratio).split("."))
    incomplete = "i" if args.keep_incomplete else "c"
    if missing_ratio == 0:
        incomplete = "c"
    if args.model_name in ["JMVAE", "JNF"]:
        incomplete = "c"
    return f"asenella/mmnist_{args.model_name}config2_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
