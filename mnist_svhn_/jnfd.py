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
args = argparse.Namespace(**info)

# Model configuration 
model_config = JNFDccaConfig(
    **base_config,
    warmup=100,
    nb_epochs_dcca=100,
    embedding_dcca_dim=9,
    beta = args.beta,
    uses_likelihood_rescaling=args.use_rescaling
)

#Architectures
dcca_networks = dict(
    mnist = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.embedding_dcca_dim,input_dim=(1,28,28))),
    svhn = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.embedding_dcca_dim, input_dim=(3,32,32)))
)


joint_encoder = MultipleHeadJointEncoder(
    torch.nn.ModuleDict(
    mnist = EncoderMNIST(num_hidden_layers=1, config=BaseAEConfig(latent_dim=model_config.latent_dim,input_dim=(1,28,28))),
    svhn = EncoderSVHN(config=BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3,32,32)))
)
)



model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders, joint_encoder=joint_encoder)

# Training configuration
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig

trainer_config = AddDccaTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}', f'seed_{args.seed}'),
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

# Validate
eval(trainer_config.output_dir, model, classifiers, wandb_cb.run.path)

# Push to HuggingFaceHub

# Push to HuggingFaceHub

model.push_to_hf_hub(f'asenella/ms_{model.model_name}_beta_{int(args.beta*10)}_scale_{args.use_rescaling}_seed_{args.seed}')



