from dataset import CUB
from multivae.models import JNF, JNFConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from torch.utils.data import random_split
from architectures_image import *
from architectures_text import *

# dataset
train_data = CUB('/home/asenella/scratch/data', split='train',max_lenght=32)

train_data, eval_data = random_split(train_data, [0.9, 0.1])


# model

model_config = JNFConfig(
    n_modalities=2,
    latent_dim=64,
    uses_likelihood_rescaling=False,
    decoders_dist=dict(image = 'laplace',
                       text ='categorical'),
    warmup=50
    
)

encoders = dict(image = EncoderImg(0,model_config.latent_dim,'normal'),
                text = Enc(0,model_config.latent_dim,'normal'))

decoders = dict(
    image = DecoderImg(model_config.latent_dim),
    text = Dec(0,model_config.latent_dim)
)


model=JNF(model_config=model_config,
                encoders = encoders,
                
                decoders=decoders,
                
                )



# trainer and callbacks

training_config = BaseTrainerConfig(
    output_dir='jnf_train',
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    num_epochs=50+model_config.warmup,
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