from dataset import CUB
from multivae.models import JNFGMC, JNFGMCConfig, GMC, GMCConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder, Encoder_VAE_MLP, BaseAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from torch.utils.data import random_split
from architectures_image import *
from architectures_text import *

# dataset
train_data = CUB('/home/asenella/scratch/data', split='train',max_lenght=32)
eval_data = CUB('/home/asenella/scratch/data', split='eval',max_lenght=32)

# GMC model

gmc_config = GMCConfig(
    n_modalities=2,
    common_dim=128,
    latent_dim=64,
    temperature=0.1
    
)

gmc_model = GMC(gmc_config,
                processors=dict(image = EncoderImg(0,gmc_config.common_dim,'normal'),
                                text = Enc(0,gmc_config.common_dim,'normal')),
                joint_encoder=MultipleHeadJointEncoder(dict(image = EncoderImg(0,gmc_config.common_dim,'normal'),
                                text = Enc(0,gmc_config.common_dim,'normal')), BaseAEConfig(latent_dim=gmc_config.common_dim)),
                shared_encoder= Encoder_VAE_MLP(BaseAEConfig(input_dim=(gmc_config.common_dim,), latent_dim=gmc_config.latent_dim))
                
                )


# model

model_config = JNFGMCConfig(
    n_modalities=2,
    latent_dim=64,
    uses_likelihood_rescaling=True,

    rescale_factors=dict(image = maxSentLen/(3*64*64),
                         text = 5.0),
    
    decoders_dist=dict(image = 'laplace',
                       text ='categorical'),
    
    decoder_dist_params=dict(image = dict(scale=0.01)),
    nb_epochs_gmc=100,
    warmup=150,
    
)

encoders = dict(
    image = Encoder_VAE_MLP(BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=model_config.latent_dim)),
    text = Encoder_VAE_MLP(BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=model_config.latent_dim))
)


decoders = dict(
    image = DecoderImg(model_config.latent_dim),
    text = Dec(0,model_config.latent_dim)
)


model=JNFGMC(model_config=model_config,
                encoders = encoders, 
                
                decoders=decoders,
                
                gmc_model=gmc_model
                
                )



# trainer and callbacks

training_config = BaseTrainerConfig(
    output_dir='jnfgmc_cub_train',
    per_device_eval_batch_size=64,
    per_device_train_batch_size=64,
    num_epochs= model_config.warmup + 60,
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