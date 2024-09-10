from multivae.models.gmc import GMC, GMCConfig
from multivae.data.datasets import MHD
from architectures import *
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from torch.utils.data import random_split

dataset = MHD('/home/asenella/scratch/data/MHD')

train_data, val_data = random_split(dataset, [0.9,0.1])

model_config = GMCConfig(
    n_modalities=4,
    input_dims = dict(image = (1,28,28), audio = (1,32,28), trajectory = (200,), label=(10,)),
    common_dim=64,
    latent_dim=64,
    temperature=0.1
)

model = GMC(
    model_config=model_config,
    processors = dict(image = MHDImageProcessor(model_config.common_dim),
                      audio = MHDSoundProcessor(model_config.common_dim),
                      trajectory= MHDTrajectoryProcessor(common_dim=model_config.common_dim),
                      label = MHDLabelProcessor(model_config.common_dim)
                      ),
    joint_encoder=MHDJointProcessor(model_config.common_dim),
    shared_encoder=MHDCommonEncoder(model_config.common_dim,model_config.latent_dim)
)

training_config = BaseTrainerConfig(
    '/home/asenella/experiments/reproduce_gmc',
    per_device_train_batch_size=64,
    num_epochs=100,
    optimizer_cls='Adam',
    learning_rate=1e-3,
)

wandb = WandbCallback()
wandb.setup(training_config,model_config,project_name='reproduce_gmc')

trainer = BaseTrainer(
    training_config=training_config,
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[wandb]
    
)

trainer.train() 