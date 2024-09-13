from dataset import CUB
from multivae.models import MMVAEPlus, MMVAEPlusConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from torch.utils.data import random_split
from architectures_image import *
from architectures_text import *

# dataset
train_data = CUB('/home/asenella/scratch/data', split='train',max_lenght=32)

train_data, eval_data = random_split(train_data, [0.9, 0.1])

# model

model_config = MMVAEPlusConfig(
    n_modalities = 2,
    latent_dim = 48,
    modalities_specific_dim=16,
    prior_and_posterior_dist='normal',
    beta=1.0,
    K=10,
    decoders_dist=dict(image = 'laplace',
                       text = 'categorical')

)


model = MMVAEPlus(model_config=model_config,
                encoders = dict(image = EncoderImg(model_config.modalities_specific_dim,model_config.latent_dim,dist='normal'),
                                text = Enc(model_config.modalities_specific_dim,model_config.latent_dim,dist='normal')),
                
                decoders=dict(
                    image = DecoderImg(model_config.latent_dim+model_config.modalities_specific_dim),
                    text = Dec(model_config.modalities_specific_dim,model_config.latent_dim)
                )
                
                )



# trainer and callbacks

training_config = BaseTrainerConfig(
    output_dir='mmvae_train',
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    num_epochs=50,
    optimizer_cls="Adam",
    optimizer_params=dict(amsgrad = True),
    learning_rate=1e-3,
    steps_predict=5,
    steps_saving=25
    
)

wandb = WandbCallback()
wandb.setup(training_config=training_config,model_config=model_config, project_name="mmvae_plus_CUB")

trainer = BaseTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    callbacks=[wandb],
    training_config=training_config
    
)

trainer.train()