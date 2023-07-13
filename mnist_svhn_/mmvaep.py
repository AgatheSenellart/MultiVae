from multivae.models import MMVAEPlus, MMVAEPlusConfig
from config import *
from multivae.models.base import BaseAEConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
from multivae.models.base import BaseEncoder, BaseDecoder

# Get the experiment configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

# Model configuration 
model_config = MMVAEPlusConfig(
    **base_config,
    beta=args.beta,
    uses_likelihood_rescaling=args.use_rescaling,
    K=1,
    modalities_specific_dim=15,
)
model_config.latent_dim = 5

class wrapper_encoder_mnist(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        self.private = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.latent_dim,input_dim=(1,28,28)))
        self.modality_specific = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.modalities_specific_dim,input_dim=(1,28,28)))
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output
    
class wrapper_encoder_svhn(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        self.private = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3,32,32)))
        self.modality_specific = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.modalities_specific_dim, input_dim=(3,32,32)))
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output



#Architectures
encoders = dict(
    image = wrapper_encoder_mnist(), 
    svhn = wrapper_encoder_svhn()
)




model = MMVAEPlus(model_config, encoders, decoders)

# Training configuration
from multivae.trainers import BaseTrainer, BaseTrainerConfig

trainer_config = BaseTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}', f'seed_{args.seed}'),
    )


train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model = model, 
    train_dataset=train, 
    eval_dataset=val,
    training_config=trainer_config, 
    callbacks=callbacks,
)

# Train 
trainer.train()
model = trainer._best_model

# Validate
eval(trainer_config.output_dir, model, classifiers, wandb_cb.run.path)

# Push to HuggingFaceHub

model.push_to_hf_hub(f'asenella/ms_{model.model_name}_beta_{int(args.beta*10)}_scale_{args.use_rescaling}_seed_{args.seed}')



