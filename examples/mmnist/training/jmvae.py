from config2 import *
from multivae.models import JMVAE, JMVAEConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=8)
parser.add_argument('--missing_ratio',type=float, default=0)
args = parser.parse_args()

train_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="train", missing_ratio=args.missing_ratio)
test_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = JMVAEConfig(
    **base_config,
    alpha = 0.1,
    warmup=200,
    
)

model = JMVAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    start_keep_best_epoch=model_config.warmup +1,
    seed=args.seed,
    output_dir= f'compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/'
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(dict(missing_ratio=args.missing_ratio))

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()
trainer._best_model.push_to_hf_hub("asenella/mmnist_{}_{}_{}".format(model.model_name, config_name,args.seed))

eval_model(model, trainer.training_dir,test_data,wandb_cb.run.path)
