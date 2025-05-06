# This is a quickstart example for using the Multivae library.

# Load a dataset
from multivae.data.datasets import MnistSvhn

train_set = MnistSvhn(data_path="./data", split="train", download=True)


# Instantiate your favorite model:
from multivae.models import MVTCAE, MVTCAEConfig

model_config = MVTCAEConfig(
    n_modalities=2,
    latent_dim=20,
    input_dims={"mnist": (1, 28, 28), "svhn": (3, 32, 32)},
)
model = MVTCAE(model_config)


# Define a trainer and train the model !
from multivae.trainers import BaseTrainer, BaseTrainerConfig

training_config = BaseTrainerConfig(learning_rate=1e-3, num_epochs=10)

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    training_config=training_config,
)
trainer.train()
