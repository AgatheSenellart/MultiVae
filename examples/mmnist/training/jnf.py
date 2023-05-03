from config2 import *

from multivae.models import JNF, JNFConfig
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=8)
parser.add_argument('--missing_ratio', default=0)
args = parser.parse_args()

train_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="train", missing_ratio=args.missing_ratio)
test_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = JNFConfig(
    **base_config,
    warmup=base_training_config["num_epochs"] // 2,
)


model = JNF(model_config, encoders=encoders, decoders=decoders)

trainer_config = TwoStepsTrainerConfig(
    **base_training_config,
    seed = args.seed,
    output_dir= f'compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/'
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

# validate the model and save

model = trainer._best_model
coherences = CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=trainer.training_dir,
).eval()

trainer._best_model.push_to_hf_hub("asenella/mmnist" + model.model_name + config_name)
