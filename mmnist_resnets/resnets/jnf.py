from config2 import *

from multivae.models import JNF, JNFConfig
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
args = parser.parse_args()


train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed)
)

model_config = JNFConfig(
    **base_config,
    warmup=100,
    latent_dim=128,
    two_steps_training=True,
    beta=1.
)

encoders = {m : Enc(ndim_w=0,ndim_u=model_config.latent_dim) for m in modalities}
decoders = {m : Dec(ndim=model_config.latent_dim) for m in modalities}

# MAYBE : try with a different joint encoder model ?

model = JNF(model_config, encoders=encoders, decoders=decoders)

trainer_config = TwoStepsTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=f"{config_name}/{model.model_name}/seed_{args.seed}",
)
trainer_config.num_epochs = 200

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model

id = [model.model_name,f'seed_{args.seed}']

save_to_hf(model, id) 
##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)
