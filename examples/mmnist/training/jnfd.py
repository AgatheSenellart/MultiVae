
from multivae.models import JNFDcca, JNFDccaConfig
from multivae.trainers import AddDccaTrainer,AddDccaTrainerConfig
from config1 import *


model_config = JNFDccaConfig(
    **base_config,
    warmup=base_training_config['num_epochs']//2,
    nb_epochs_dcca=200,
    embedding_dcca_dim=20,
)

dcca_networks = {
    k: encoder_class(
        BaseAEConfig(latent_dim=model_config.embedding_dcca_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders)

trainer_config = AddDccaTrainerConfig(
    **base_training_config,
    learning_rate_dcca=1e-4,
    per_device_dcca_eval_batch_size=800
)
trainer_config.num_epochs += model_config.nb_epochs_dcca # Add the DCCA time to overall number of epochs

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

# validate the model and save

model = trainer._best_model
coherences = CoherenceEvaluator(model=model,
                                test_dataset=test_data,
                                classifiers=load_mmnist_classifiers(device=model.device),
                                output=trainer.training_dir)

trainer._best_model.push_to_hf_hub('asenella/mmnist'+ model.model_name + config_name)