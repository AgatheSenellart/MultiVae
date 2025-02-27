from global_config import *

from multivae.models import MoPoE, MoPoEConfig

args = argument_parser().parse_args()

# Define datasets
train_data, eval_data, test_data = get_datasets(args)

# Define model configuration: add hyperparameters specific to the model
model_config = MoPoEConfig(beta=2.5, **base_config)

model = MoPoE(model_config, encoders=encoders, decoders=decoders)

# Define the training configuration
trainer_config = BaseTrainerConfig(
    **base_training_config,
    drop_last=True, # specific to MoPoE, works best
    seed=args.seed,
    output_dir=model_save_path(model, args)
)
trainer_config.num_epochs = 400

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=WANDB_PROJECT)
wandb_cb.run.config.update(args.__dict__)

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=[wandb_cb],
)

# Train the model
trainer.train()

# Get best model and perform evaluation 
model = trainer._best_model

eval_model(model, trainer.training_dir, test_data, wandb_cb.run.path)
