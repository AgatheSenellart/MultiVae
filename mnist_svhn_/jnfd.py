from multivae.models import JNFDccaConfig, JNFDcca
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
model_config = JNFDccaConfig(
    **base_config,
    warmup=100,
    nb_epochs_dcca=100,
    **args,
)

#Architectures
dcca_networks = dict(
    mnist = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.embedding_dcca_dim,input_dim=(1,28,28))),
    svhn = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.embedding_dcca_dim, input_dim=(3,32,32)))
)


joint_encoder = MultipleHeadJointEncoder(
    dict(
    mnist = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.latent_dim,input_dim=(1,28,28))),
    svhn = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3,32,32)))
),
    args=BaseAEConfig(latent_dim=model_config.latent_dim)
)



model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders, joint_encoder=joint_encoder)

id = [(f'{m}_{int(args[m]*100)}' if (type(args[m])==float) else f'{m}_{args[m]}') for m in args]

# Training configuration
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig

trainer_config = AddDccaTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, *id),
    per_device_dcca_train_batch_size=800,
    per_device_dcca_eval_batch_size=800,
    learning_rate_dcca=1e-3
    )

trainer_config.num_epochs += model_config.nb_epochs_dcca

train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

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
save_to_hf(model, id)


# Validate
eval(trainer_config.output_dir, model, classifiers, wandb_cb.run.path)






