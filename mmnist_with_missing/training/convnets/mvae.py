from config2 import *

from multivae.models import MVAE, MVAEConfig

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train",
    missing_ratio=args.missing_ratio,
    keep_incomplete=args.keep_incomplete,
)
test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = MVAEConfig(
    **base_config,
    use_subsampling=(args.missing_ratio == 0) or not (args.keep_incomplete),
    # use_subsampling=True,
    warmup=0,
    beta=2.5,
)


model = MVAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=f"compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/",
    start_keep_best_epoch=model_config.warmup + 1,
)
trainer_config.num_epochs = 500
trainer_config.learning_rate = 5e-4
trainer_config.drop_last = True
trainer_config.scheduler_cls = None
trainer_config.scheduler_params = None

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model
save_model(model, args)

##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir, test_data, wandb_cb.run.path)
