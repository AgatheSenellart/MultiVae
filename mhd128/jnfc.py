from multivae.models import JNFCLIPConfig, JNFCLIP
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
args = info

# Model configuration 
model_config = JNFCLIPConfig(
    **base_config,
    warmup=200,
    nb_epochs_clip=50,
    **args
    
)

#Architectures
clip_networks = dict(
    image = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = model_config.embedding_clip_dim)), 
    audio = SoundEncoder(model_config.embedding_clip_dim),
    trajectory = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=model_config.embedding_clip_dim)
)

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



model = JNFCLIP(model_config, CLIP_networks=clip_networks, decoders=decoders, joint_encoder=joint_encoder)

# Training configuration
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
id = [(f'{m}_{int(args[m]*100)}' if (type(args[m])==float) else f'{m}_{args[m]}') for m in args]


trainer_config = AddDccaTrainerConfig(
    **base_trainer_config,
    seed=args['seed'],
    output_dir=os.path.join(project_path, model.model_name, *id),
    per_device_dcca_train_batch_size=256,
    per_device_dcca_eval_batch_size=256,
    learning_rate_dcca=1e-3
    )

trainer_config.num_epochs += model_config.nb_epochs_clip

train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args['seed']))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = AddDccaTrainer(
    model=model, 
    train_dataset = train, 
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






