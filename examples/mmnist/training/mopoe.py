
from multivae.models import MoPoE, MoPoEConfig
from config2 import *

model_config = MoPoEConfig(
    beta=2.5,
    **base_config
)


model = MoPoE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    drop_last=True
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

model = trainer._best_model
# validate the model
coherences = CoherenceEvaluator(model=model,
                                test_dataset=test_data,
                                classifiers=load_mmnist_classifiers(device=model.device),
                                output=trainer.training_dir).eval()

trainer._best_model.push_to_hf_hub('asenella/mmnist'+ model.model_name + config_name)
