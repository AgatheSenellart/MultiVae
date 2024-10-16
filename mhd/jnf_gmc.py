from multivae.models import JNFGMC, JNFGMCConfig, GMC, GMCConfig
from config import *
from multivae.models.base import BaseAEConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
import torch

# Get the experiment configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)


# GMC configuration
gmc_config = GMCConfig(
    n_modalities=3,
    input_dims=base_config['input_dims'],
    common_dim=16,
    latent_dim=16,
    temperature=args.temperature,
    loss=args.gmc_loss
)

# GMC architectures and model
gmc_encoders = dict(
    image = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = gmc_config.common_dim)), 
    audio = SoundEncoder(gmc_config.common_dim),
    trajectory = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=gmc_config.common_dim)
)

class MHDCommonEncoder(BaseEncoder):
    def __init__(self, common_dim, latent_dim):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128),
            Swish(),
            nn.Linear(128, latent_dim),
        )
    def forward(self, x):
        return ModelOutput(embedding = F.normalize(self.feature_extractor(x), dim=-1))

joint_encoder_gmc = None if gmc_config.loss == 'between_modality_pairs' else MultipleHeadJointEncoder(gmc_encoders, BaseAEConfig(latent_dim=gmc_config.common_dim))

gmc_model = GMC(
    model_config=gmc_config,
    shared_encoder=MHDCommonEncoder(gmc_config.common_dim, gmc_config.latent_dim),
    processors=gmc_encoders,
    joint_encoder = joint_encoder_gmc,
)
# Model configuration 
model_config = JNFGMCConfig(
    **base_config,
    warmup=args.warmup,
    annealing=args.annealing,
    nb_epochs_gmc=100,
    beta=args.beta,
    logits_to_std='standard',
    uses_likelihood_rescaling=args.use_rescaling
)


# Model architectures and instanciation

decoders = dict(
    image = Decoder_Conv_AE_MNIST(BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3,28,28))),
    audio = SoundDecoder(model_config.latent_dim),
    trajectory = TrajectoryDecoder(model_config.latent_dim, [512,512,512],output_dim=200)
)

joint_encoder = MultipleHeadJointEncoder(
    dict(
    image = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = model_config.latent_dim)), 
    audio = SoundEncoder(model_config.latent_dim),
    trajectory = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.latent_dim)
),
    args = BaseAEConfig(latent_dim=model_config.latent_dim)
)

model = JNFGMC(model_config,gmc_model=gmc_model,decoders=decoders,joint_encoder=joint_encoder)


# Training configuration
from multivae.trainers import MultistageTrainer, MultistageTrainerConfig

trainer_config = MultistageTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}'),
    )

trainer_config.num_epochs = model_config.nb_epochs_gmc + model_config.warmup + 100

train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name='look_for_best_jnf_gmc_parameters')
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = MultistageTrainer(
    model=model, 
    train_dataset = train, 
    eval_dataset=val, 
    training_config=trainer_config, 
    callbacks=callbacks,
)

# Train 
trainer.train()
model = trainer._best_model

wandb_id = wandb_cb.run._run_id.replace('-','_')

# Push to HuggingFaceHub
model.push_to_hf_hub(f'asenella/MHD_{model.model_name}_{wandb_id}')

# Validate
eval(trainer_config.output_dir, model, classifiers, wandb_cb.run.path)






