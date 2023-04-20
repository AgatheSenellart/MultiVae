import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets import MMNISTDataset
from multivae.models import JNF, JNFConfig
from multivae.models.nn.mmnist import DecoderImg, EncoderImg
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MMNISTDataset(data_path="../../../data/MMNIST", split="train")
train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)
print(len(train_data), len(eval_data))
modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = JNFConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=512,
    warmup=300,
)

modalities

encoders = {
    k: EncoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: DecoderImg(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = JNF(model_config, encoders=encoders, decoders=decoders)

print(model.reset_optimizer_epochs)


trainer_config = TwoStepsTrainerConfig(
    num_epochs=600,
    learning_rate=1e-4,
    steps_predict=1,
    per_device_train_batch_size=256,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
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
