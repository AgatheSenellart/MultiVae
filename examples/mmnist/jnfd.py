import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.models import JNFDcca, JNFDccaConfig
from multivae.models.nn.mmnist import Decoder_ResNet_AE_MNIST, Encoder_ResNet_VAE_MMNIST
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MMNISTDataset(data_path="../../../data/MMNIST", split="train")
train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)
print(len(train_data))
modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = JNFDccaConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=512,
    nb_epochs_dcca=200,
    warmup=200,
    use_likelihood_rescaling=True,
    decoders_dist={k: "laplace" for k in modalities},
    embedding_dcca_dim=20,
)

modalities

dcca_networks = {
    k: Encoder_ResNet_VAE_MMNIST(BaseAEConfig(latent_dim=20, input_dim=(3, 28, 28)))
    for k in modalities
}

decoders = {
    k: Decoder_ResNet_AE_MNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = JNFDcca(
    model_config, encoders=None, decoders=decoders, dcca_networks=dcca_networks
)

trainer_config = AddDccaTrainerConfig(
    num_epochs=200 + 200 + 200,
    learning_rate=1e-3,
    learning_rate_dcca=1e-4,
    steps_predict=1,
    per_device_dcca_train_batch_size=800,
    per_device_train_batch_size=256,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = AddDccaTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

# data = set_inputs_to_device(eval_data[:100], device="cuda")
# nll = model.compute_joint_nll(data)
# print(nll)
