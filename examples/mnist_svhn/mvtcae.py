from config1 import *

from multivae.models import MVTCAE, MVTCAEConfig

model_config = MVTCAEConfig(beta=2.5, alpha=2.0 / 3.0, **base_model_config)


model = MVTCAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(**base_training_config)

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
coherences = CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mnist_svhn_classifiers(data_path_classifiers, device=model.device),
    output=trainer.training_dir,
).eval()

trainer._best_model.push_to_hf_hub("asenella/ms" + model.model_name + config_name)
