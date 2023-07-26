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
    warmup=100,
    beta=args.beta,
    uses_likelihood_rescaling=args.use_rescaling
)




model = JNF(model_config, encoders=encoders, decoders=decoders)

# Training configuration
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig

trainer_config = TwoStepsTrainerConfig(
    **base_trainer_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, f'beta_{int(args.beta*10)}', f'rescale_{args.use_rescaling}', f'seed_{args.seed}'),
    )


train, val = random_split(train_set, [0.9,0.1], generator=torch.Generator().manual_seed(args.seed))



# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
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

model.push_to_hf_hub(f'asenella/ms_{model.model_name}_beta_{int(args.beta*10)}_scale_{args.use_rescaling}_seed_{args.seed}')



