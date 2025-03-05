from global_config import *

from multivae.models import MVAE, MVAEConfig

# Get parameters for the experiments
args = argument_parser().parse_args()

# Define datasets
train_data, eval_data, test_data = get_datasets(args)

# Define model configuration: add hyperparameters specific to the model
model_config = MVAEConfig(
    **base_config,
    use_subsampling=(args.missing_ratio == 0) or not (args.keep_incomplete),
    # use_subsampling=True,
    warmup=0,
    beta=2.5,
)

model = MVAE(model_config, encoders=encoders, decoders=decoders)

# Define the training configuration
trainer_config = BaseTrainerConfig(
    **base_training_config, seed=args.seed, output_dir=model_save_path(model, args)
)
# change parameters to avoid crashing
trainer_config.num_epochs = 500
trainer_config.learning_rate = 5e-4
trainer_config.drop_last = True

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
