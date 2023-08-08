from multivae.models import JNFConfig, JNF
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
model_config = JNFConfig(
    **base_config,
    warmup=100
)

#Architectures
encoders = dict(
    image = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = model_config.latent_dim)), 
    audio = SoundEncoder(model_config.latent_dim),
    trajectory = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.latent_dim)
)

decoders = dict(
    image = Decoder_Conv_AE_MNIST(BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3,28,28))),
    audio = SoundDecoder(model_config.latent_dim),
    trajectory = TrajectoryDecoder(model_config.latent_dim, [512,512,512],output_dim=200)
)


model = JNF(model_config, encoders, decoders)

# Training configuration
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig

trainer_config = TwoStepsTrainerConfig(
    **base_trainer_config,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}'),
    )

trainer_config.num_epochs = 1

train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model, 
    val, 
    val, trainer_config, 
    callbacks=callbacks,
)

# Train 
trainer.train()
model = trainer._best_model

# Push to HuggingFaceHub

from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=os.path.join(trainer.training_dir, 'final_model'),
    path_in_repo=f'{model.model_name}/beta_{int(args.beta*10)}/rescale_{args.use_rescaling}/seed_{args.seed}', # Upload to a specific folder
    repo_id="asenella/test_repository",
    repo_type="model",
)


