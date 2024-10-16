from dataset import CUB
from multivae.models import MMVAEPlus, MMVAEPlusConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from torch.utils.data import random_split
from architectures_image import *
from architectures_text import *
from utils import *
from torch.utils.data import random_split

# dataset
train_data = CUB(data_path, split='train',max_lenght=32)
eval_data = CUB(data_path, split='eval',max_lenght=32)

# model

model_config = MMVAEPlusConfig(
    n_modalities = 2,
    latent_dim = 48,
    modalities_specific_dim=16,
    prior_and_posterior_dist='normal_with_softplus',
    beta=1.0,
    K=10,
    decoders_dist=dict(image = 'laplace',
                       text = 'categorical'),
    
    decoder_dist_params=dict(image = dict(scale=0.01)),
    
    uses_likelihood_rescaling=True,
    rescale_factors=dict(image = maxSentLen/(3*64*64),
                         text = 5.0),
    
    learn_shared_prior=False,
    learn_modality_prior=True,
    loss = 'dreg_looser'
    

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
    output_dir=save_path,
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    num_epochs=50,
    optimizer_cls="Adam",
    optimizer_params=dict(amsgrad = True),
    learning_rate=1e-3,
    steps_predict=5,
    seed=2
    
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

trainer._best_model.push_to_hf_hub(f'asenella/{CUB}_{model.model_name}_{wandb.run._name}')

# Validate and compute coherence
from evaluate_coherence import evaluate_coherence
test_data = CUB(data_path, split='test',max_lenght=32).text_data
model = trainer._best_model
wandb_path = wandb.run._get_path()
evaluate_coherence(model, wandb_path,test_data)