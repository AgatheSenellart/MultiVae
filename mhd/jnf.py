from multivae.models import JNFConfig, JNF
from config import *
from multivae.models.base import BaseAEConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str)
args = parser.parse_args()

model_config = JNFConfig(
    **base_config,
    warmup=200
)

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

from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig

trainer_config = TwoStepsTrainerConfig(
    **base_trainer_config,
    output_dir=os.path.join(project_path, model_config.name)
    )
    

train, val = random_split(train_set, [0.8,0.2], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model, 
    test_set, 
    val, trainer_config, 
    callbacks=callbacks
)


trainer.train()

eval(trainer_config.output_dir, model, classifiers)

