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
args = info


# Model configuration 
model_config = MMVAEPlusConfig(
    **base_config,
    **args,
    learn_shared_prior=False,
    K=10,
    modalities_specific_dim=64
)
model_config.latent_dim = 64

class wrapper_encoder_image(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        self.private = Encoder_Conv_VAE_MNIST(BaseAEConfig((1,28,28), latent_dim = model_config.latent_dim))
        self.modality_specific = Encoder_Conv_VAE_MNIST(BaseAEConfig((1,28,28), latent_dim = model_config.modalities_specific_dim))
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output
    
class wrapper_encoder_sound(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        self.private = SoundEncoder(model_config.latent_dim)
        self.modality_specific = SoundEncoder(model_config.modalities_specific_dim)
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output

class wrapper_encoder_traj(BaseEncoder):
    
    def __init__(self):
        super().__init__()
        self.private = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.latent_dim)
        self.modality_specific = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.modalities_specific_dim)
    
    def forward(self,input):
        output = self.private(input)
        output_modality_specific = self.modality_specific(input)
        output['style_embedding'] = output_modality_specific['embedding']
        output['style_log_covariance'] = output_modality_specific['log_covariance']
        return output

#Architectures
encoders = dict(
    image = wrapper_encoder_image(), 
    audio = wrapper_encoder_sound(),
    trajectory = wrapper_encoder_traj()
)

decoders = dict(
    image = Decoder_Conv_AE_MNIST(BaseAEConfig(latent_dim=model_config.latent_dim+model_config.modalities_specific_dim)),
    audio = SoundDecoder(model_config.latent_dim+model_config.modalities_specific_dim),
    trajectory = TrajectoryDecoder(model_config.latent_dim+model_config.modalities_specific_dim, [512,512,512],output_dim=200)
)


model = MMVAEPlus(model_config, encoders, decoders)

# Training configuration
from multivae.trainers import BaseTrainer, BaseTrainerConfig

id = [(f'{m}_{int(args[m]*100)}' if (type(args[m])==float) else f'{m}_{args[m]}') for m in args]


trainer_config = BaseTrainerConfig(
    **base_trainer_config,
    seed=args['seed'],
    output_dir=os.path.join(project_path, model.model_name, *id),
    )
if model_config.K == 10:
    trainer_config.num_epochs = 100

train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args['seed']))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args)

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

# Push to HuggingFaceHub
save_to_hf(model, id)

# Validate
eval(trainer_config.output_dir, model, classifiers, wandb_cb.run.path)


