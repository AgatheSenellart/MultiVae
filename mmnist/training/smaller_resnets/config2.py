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

modalities = ["m0", "m1", "m2", "m3", "m4"]

base_config = dict(
    n_modalities=len(modalities),
    input_dims={k: (3, 28, 28) for k in modalities},
    decoders_dist={k: "laplace" for k in modalities},
    decoder_dist_params={k: {"scale": 0.75} for k in modalities},
)

###### Architectures ######

from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput
from multivae.models.nn.mmnist import Encoder_ResNet_VAE_MMNIST, Decoder_ResNet_AE_MMNIST

class Enc(BaseEncoder):
    
    def __init__(self, ndim_w, ndim_u):
        
        super().__init__()
        self.latent_dim = ndim_u
        
        self.shared_encoder = Encoder_ResNet_VAE_MMNIST(BaseAEConfig(
            input_dim=(3,28,28),
            latent_dim=ndim_u
        ))
        
        if ndim_w > 0:
            self.multiple_latent_spaces = True
            self.private_encoder = Encoder_ResNet_VAE_MMNIST(BaseAEConfig(
            input_dim=(3,28,28),
            latent_dim=ndim_u
        ))
        else:
            self.multiple_latent_spaces = False
            
    
    def forward(self, x):
        
        out = self.shared_encoder(x)
        if self.multiple_latent_spaces:
            out_private = self.private_encoder(x)
            out['style_embedding'] = out_private['embedding']
            out['style_log_covariance'] = out_private['log_covariance']
        return out
    
class Dec(BaseDecoder):
    
    def __init__(self,ndim):
        super().__init__()
        self.decoder = Decoder_ResNet_AE_MMNIST(BaseAEConfig(latent_dim=ndim, input_dim=(3,28,28)))
        self.latent_dim = ndim
        
    def forward(self, x):
        return self.decoder(x)


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

wandb_project = "compare_on_mmnist_resnet"
config_name = "config_smaller_resnets"


#######################################
## Define parameters for the evaluation


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h


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


def save_model(model, args):
    missing_ratio = "".join(str(args.missing_ratio).split("."))
    incomplete = "i" if args.keep_incomplete else "c"
    model.push_to_hf_hub(
        f"asenella/mmnist_{model.model_name}{config_name}_seed_{args.seed}_ratio_{missing_ratio}_{incomplete}"
    )


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
        
    
