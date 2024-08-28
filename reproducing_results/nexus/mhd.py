##############################################################
# In this file, we reproduce the experiments from the Nexus paper :
#      "Leveraging hierarchy in multimodal generative models for effective cross-modality inference" (Vasco et al 2022)


from multivae.models import Nexus, NexusConfig
from multivae.data.datasets.mhd import MHD
from multivae.models.base import BaseEncoder, BaseAEConfig, BaseDecoder, ModelOutput
from torch import nn 
import torch
from math import prod
from torch.utils.data import  random_split
import torch.nn.functional as F
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    WandbCallback,
)
from architectures import *
from image_architectures import *
from trajectory_architectures import *
from sound_architectures import *
from symbol_architectures import *

# import the datasets
train_set = MHD('/home/asenella/scratch/data/MHD', split='train', modalities=['audio', 'trajectory', 'image', 'label'])
test_set = MHD('/home/asenella/scratch/data/MHD', split='test', modalities=['audio', 'trajectory', 'image', 'label'])

train_set, eval_set = random_split(train_set,[0.9, 0.1])



# Model config

model_config = NexusConfig(
    n_modalities=4,
    input_dims = dict(image = (1,28,28), audio = (1,32,28), trajectory = (200,), label=(10,)),
    modalities_specific_dim=dict(image = 64, audio = 128, label = 5, trajectory = 16),
    latent_dim=32,
    msg_dim=512,
    gammas= dict(image = 1.0, trajectory = 50.0, audio = 1.0, label= 50.0),
    bottom_betas=dict(image = 1.0,trajectory = 1.0,audio = 1.0,label = 1.0),
    uses_likelihood_rescaling=True,
    rescale_factors=dict(image = 1.0, trajectory=50.0, audio = 1.0, label=50.0),
    top_beta=1.0,
    warmup = 20,
    dropout_rate=0.2,
    decoders_dist=dict(image = 'normal', audio='normal',trajectory = 'normal',label='bernoulli'),
    )


model = Nexus(model_config,
              encoders = dict(image = ImageEncoder('image_encoder', 28, 1, [32,64], [128,128], model_config.modalities_specific_dim['image']),
                              audio = SoundEncoder(model_config.modalities_specific_dim['audio']),
                              trajectory = TrajectoryEncoder('traj_encoder', 200, [512, 512, 512], model_config.modalities_specific_dim['trajectory']),
                              label = SymbolEncoder('symbol_encoder', 10, [128, 128, 128], model_config.modalities_specific_dim['label'])
                              ),
              decoders=dict(image = ImageDecoder('image_decoder', model_config.modalities_specific_dim['image'], 1, [64,32], [128,128], 28),
                            audio = SoundDecoder(model_config.modalities_specific_dim['audio']),
                            trajectory = TrajectoryDecoder( 'traj_decoder', model_config.modalities_specific_dim['trajectory'], [512, 512, 512], 200),
                            label = SymbolDecoder('symbol_encoder', model_config.modalities_specific_dim['label'], [128, 128, 128], 10)
                            ),
              top_encoders=dict(image = TopEncoder(model_config.modalities_specific_dim['image'],model_config.msg_dim),
                                audio = TopEncoder(model_config.modalities_specific_dim['audio'],model_config.msg_dim),
                                trajectory=TopEncoder(model_config.modalities_specific_dim['trajectory'],model_config.msg_dim),
                                label=TopEncoder(model_config.modalities_specific_dim['label'],model_config.msg_dim),
                                ),
              
              top_decoders =dict(image = Topdecoder(model_config.modalities_specific_dim['image'],model_config.latent_dim),
                                audio = Topdecoder(model_config.modalities_specific_dim['audio'],model_config.latent_dim),
                                trajectory=Topdecoder(model_config.modalities_specific_dim['trajectory'],model_config.latent_dim),
                                label=Topdecoder(model_config.modalities_specific_dim['label'],model_config.latent_dim),
                                ),
              
              joint_encoder=JointEncoder(model_config.msg_dim, model_config.latent_dim)
              
              )


# print(model(next(iter(DataLoader(train_set,2)))))


########## Training #######
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig

training_config = BaseTrainerConfig(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=100,
    learning_rate=1e-3,
    steps_predict=5,
    optimizer_cls="Adam",
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_nexus")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=eval_set,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

trainer._best_model.push_to_hf_hub(
    f"asenella/reproduce_nexus"
)
