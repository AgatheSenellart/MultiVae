from config2 import *
from multivae.models import JMVAE, JMVAEConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,default=0)
args = parser.parse_args()

model_config = JMVAEConfig(
    **base_config,
    alpha = 0.1,
    warmup=200,
    
)


model = JMVAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    start_keep_best_epoch=model_config.warmup +1,
    seed=args.seed
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)

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
