"""
Store in this file all the shared variables for the benchmark on mmnist.
"""

import argparse
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, Visualization, VisualizationConfig
from multivae.metrics.base import EvaluatorConfig
from multivae.metrics.fids.fids import FIDEvaluator
from multivae.metrics.fids.fids_config import FIDEvaluatorConfig
from multivae.models import BaseMultiVAEConfig
from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from multivae.trainers import BaseTrainerConfig
from multivae.trainers.base.base_trainer import BaseTrainer
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
from architectures import Enc, Dec, ClfImg
import os



modalities = ["m0", "m1", "m2", "m3", "m4"]


project_path = '/home/asenella/scratch/mmnist_resnet'
wandb_project = "mmnist_resnet"
config_name = "mmnist_resnet"

base_config = dict(
    n_modalities=len(modalities),
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
    decoder_dist_params={k: {"scale": 0.75} for k in modalities},
)



base_training_config = dict(
    learning_rate=1e-3,
    per_device_train_batch_size=128,
    num_epochs=500,
    optimizer_cls="Adam",
    optimizer_params={},
    steps_predict=5,
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 30},
)



def load_mmnist_classifiers(data_path="/home/asenella/scratch/data/clf", device="cuda"):
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


def eval_model(model, output_dir, train_data,test_data, wandb_path, seed):
    """
    In this function, define all the evaluation metrics
    you want to use
    """
    
    # Coherence evaluator
    config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_path)
    mod = CoherenceEvaluator(
        model=model,
        test_dataset=test_data,
        classifiers=load_mmnist_classifiers(device=model.device),
        output=output_dir,
        eval_config=config,
    )
    mod.eval()
    mod.finish()

    # FID evaluator
    config = FIDEvaluatorConfig(batch_size=128, wandb_path=wandb_path)

    fid = FIDEvaluator(
        model, test_data, output=output_dir, eval_config=config
    )
    fid.compute_all_conditional_fids(gen_mod="m0")
    fid.finish()
    
    # Visualization
    if seed == 0:
    # visualize some unconditional sample from prior
        vis_config = VisualizationConfig(wandb_path = wandb_path,n_samples=8, n_data_cond=10)

        vis_module = Visualization(model, test_data,eval_config=vis_config,output = output_dir)
        vis_module.eval()

        # And some conditional samples too
        for i in range(2,5):
            subset = modalities[1:1+i]
            vis_module.conditional_samples_subset(subset)

        vis_module.finish()
        
    # Gaussian Sampler
    from multivae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

    sampler = GaussianMixtureSampler(model)
    sampler.fit(train_data)
    config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_path)

    module_eval = CoherenceEvaluator(model,load_mmnist_classifiers(),test_data,eval_config=config,sampler=sampler)
    module_eval.joint_coherence()
    module_eval.log_to_wandb()
    module_eval.finish()

    if seed == 0:
        vis_module = Visualization(model, test_data,eval_config=vis_config,output = output_dir, sampler=sampler)
        vis_module.eval()
        vis_module.finish()
        
def save_to_hf(model, id):
    model.push_to_hf_hub(
        f'asenella/{config_name}_{"_".join(id)}')
