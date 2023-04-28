from config1 import *

from multivae.models import JNFDcca, JNFDccaConfig
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig

model_config = JNFDccaConfig(
    **base_model_config, warmup=100, nb_epochs_dcca=100, embedding_dcca_dim=20
)


model = JNFDcca(model_config, dcca_networks=encoders, decoders=decoders)

trainer_config = AddDccaTrainerConfig(
    **base_training_config,
    per_device_dcca_train_batch_size=800,
    per_device_dcca_eval_batch_size=800,
    learning_rate=1e-3,
)
trainer_config.num_epochs += model_config.nb_epochs_dcca

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = AddDccaTrainer(
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
