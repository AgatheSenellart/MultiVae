from global_config import *

from multivae.models import JNF, JNFConfig
from multivae.trainers import MultistageTrainer, MultistageTrainerConfig

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)


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
    warmup=200,
    latent_dim=args.latent_dim,
    two_steps_training=args.two_steps_training,
    beta=args.beta
)

encoders = {m : Enc(ndim_w=0,ndim_u=model_config.latent_dim) for m in modalities}
decoders = {m : Dec(ndim=model_config.latent_dim) for m in modalities}

# MAYBE : try with a different joint encoder model ?

model = JNF(model_config, encoders=encoders, decoders=decoders)

trainer_config = MultistageTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=f"~/experiments/mmnist_resnets/{model.model_name}/seed_{args.seed}/",
)
trainer_config.num_epochs = model_config.warmup + 200

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name='hyperparameter_search_jnf_mmnist_resnets')
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = MultistageTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model


##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)

save_to_hf(model, wandb_cb) 