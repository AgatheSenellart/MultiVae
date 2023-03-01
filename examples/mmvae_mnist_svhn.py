import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import DataLoader, random_split

from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.utils import save_all_images
from multivae.data.utils import set_inputs_to_device
from multivae.models import MMVAE, MMVAEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

train_data = MnistSvhn(split="test")
train_data, eval_data = random_split(
    train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

model_config = MMVAEConfig(
    n_modalities=2,
    input_dims=dict(mnist=(1, 28, 28), svhn=(3, 32, 32)),
    latent_dim=20,
    use_likelihood_rescaling=True,
    K=30,
)

encoders = dict(
    mnist=Encoder_VAE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Encoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

decoders = dict(
    mnist=Decoder_AE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1, 28, 28))),
    svhn=Decoder_VAE_SVHN(BaseAEConfig(latent_dim=20, input_dim=(3, 32, 32))),
)

model = MMVAE(model_config, encoders, decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=60,
    learning_rate=1e-3,
    steps_predict=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config=model_config, project_name="package")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
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