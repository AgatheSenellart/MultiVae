from multivae.models import MoPoE, MoPoEConfig
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
model_config = MoPoEConfig(
    **base_config,
    beta=args.beta,
    uses_likelihood_rescaling=args.use_rescaling,
    
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


model = MoPoE(model_config, encoders, decoders)

# Training configuration
from multivae.trainers import BaseTrainer, BaseTrainerConfig

trainer_config = BaseTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}'),
    drop_last=True
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

model.push_to_hf_hub(f'asenella/{model.model_name}_beta_{int(args.beta*10)}_scale_{args.use_rescaling}_seed_{args.seed}')



