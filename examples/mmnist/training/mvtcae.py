from multivae.metrics.fids.fids import FIDEvaluator
from config2 import *

from multivae.models import MVTCAE, MVTCAEConfig

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=8)
parser.add_argument('--missing_ratio', default=0)
args = parser.parse_args()

train_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="train", missing_ratio=args.missing_ratio)
test_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = MVTCAEConfig(beta=2.5, alpha=5.0 / 6.0, **base_config)


model = MVTCAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir= f'compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/'
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb._wandb.config.add(dict(missing_ratio=args.missing_ratio))

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
model.push_to_hf_hub(f"asenella/mmnist_{model.model_name}{config_name}_seed_{args.seed}_ratio_{args.missing_ratio}")

##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

coherences = CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=trainer.training_dir,
).eval()

fids = FIDEvaluator(model,
                    test_data,
                    output=trainer.training_dir,
                    ).mvtcae_reproduce_fids(gen_mod='m0')

