from multivae.models import Nexus, NexusConfig
from config import *
from multivae.models.base import BaseAEConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

# Get the experiment configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

# Model configuration 
model_config = NexusConfig(
    **base_config,
    uses_likelihood_rescaling=args.use_rescaling,
    modalities_specific_dim=dict(image = 64, audio = 128, trajectory = 16),
    msg_dim=512,
    gammas= dict(image = 1.0, trajectory = 50.0, audio = 1.0),
    bottom_betas=dict(image = 1.0,trajectory = 1.0,audio = 1.0),
    rescale_factors=dict(image = 1.0, trajectory=50.0, audio = 1.0),
    top_beta=args.top_beta,
    warmup = 20,
    dropout_rate=0.2
    
)

#Architectures
encoders = dict(
    image = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = model_config.modalities_specific_dim['image'])), 
    audio = SoundEncoder(model_config.modalities_specific_dim['audio']),
    trajectory = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.modalities_specific_dim['trajectory'])
)

decoders = dict(
    image = Decoder_Conv_AE_MNIST(BaseAEConfig(latent_dim=model_config.modalities_specific_dim['image'], input_dim=(3,28,28))),
    audio = SoundDecoder(model_config.modalities_specific_dim['audio']),
    trajectory = TrajectoryDecoder(model_config.modalities_specific_dim['trajectory'], [512,512,512],output_dim=200)
)

# Top encoders

class TopEncoder(BaseEncoder):
    
    def __init__(self, input_dim, msg_dim):
        super().__init__()
        
        self.layers = nn.Linear(input_dim,msg_dim)
    def forward(self, z):
        return ModelOutput(embedding = self.layers(z), log_covariance = None)


# Joint encoder

class JointEncoder(BaseEncoder):
    
    def __init__(self, msg_dim, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(msg_dim,512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512,512),nn.BatchNorm1d(512),nn.LeakyReLU()
        )
        self.fc_embedding = nn.Linear(512,latent_dim)
        self.fc_covariance = nn.Linear(512,latent_dim)
        
    def forward(self, z):
        h = self.layers(z)
        return ModelOutput(
            embedding = self.fc_embedding(h),
            log_covariance = self.fc_covariance(h)
        )

# Top decoders

class Topdecoder(BaseDecoder):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, input_dim)
            

        )

    def forward(self, z):
        
        return ModelOutput(reconstruction = self.layers(z))


top_encoders=dict(image = TopEncoder(model_config.modalities_specific_dim['image'],model_config.msg_dim),
                                audio = TopEncoder(model_config.modalities_specific_dim['audio'],model_config.msg_dim),
                                trajectory=TopEncoder(model_config.modalities_specific_dim['trajectory'],model_config.msg_dim),
                                )

top_decoders =dict(image = Topdecoder(model_config.modalities_specific_dim['image'],model_config.latent_dim),
                                audio = Topdecoder(model_config.modalities_specific_dim['audio'],model_config.latent_dim),
                                trajectory=Topdecoder(model_config.modalities_specific_dim['trajectory'],model_config.latent_dim),
                                )

joint_encoder=JointEncoder(model_config.msg_dim, model_config.latent_dim)


model = Nexus(
    model_config=model_config,
    encoders=encoders,
    decoders=decoders,
    top_decoders=top_decoders,
    top_encoders=top_encoders,
    joint_encoder=joint_encoder
)

# Training configuration
from multivae.trainers import BaseTrainer, BaseTrainerConfig

trainer_config = BaseTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.top_beta*10)}', f'rescale_{args.use_rescaling}'),
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

wandb_id = wandb_cb.run._run_id.replace('-','_')

model.push_to_hf_hub(f'asenella/MHD_{model.model_name}_{wandb_id}')


