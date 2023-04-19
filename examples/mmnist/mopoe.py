import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import DataLoader, random_split

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.data.datasets.utils import save_all_images
from multivae.data.utils import set_inputs_to_device
from multivae.models import MoPoE, MoPoEConfig
from multivae.models.nn.default_architectures import Decoder_AE_MLP, Encoder_VAE_MLP
from multivae.models.nn.mmnist import Decoder_ResNet_AE_MNIST, Encoder_ResNet_VAE_MMNIST
from multivae.models.nn.svhn import Decoder_VAE_SVHN, Encoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

# train_data = MMNISTDataset(data_path="../../../data/MMNIST", split="train")
# train_data, eval_data = random_split(
#     train_data, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
# )
# print(len(train_data), len(eval_data))

modalities = ["m0", "m1", "m2", "m3", "m4"]

model_config = MoPoEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=128,
    decoders_dist={m: "laplace" for m in modalities},
    beta=2.5,
    decoder_scale=0.75,
)


encoders = {
    k: Encoder_ResNet_VAE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: Decoder_ResNet_AE_MNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = MoPoE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=1e-3,
    steps_predict=1,
    per_device_train_batch_size=256,
    drop_last=True,
)

# Set up callbacks
# wandb_cb = WandbCallback()
# wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

# callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

# trainer = BaseTrainer(
#     model,
#     train_dataset=train_data,
#     eval_dataset=eval_data,
#     training_config=trainer_config,
#     callbacks=callbacks,
# )
# trainer.train()

model.push_to_hf_hub("asenella/test")

from multivae.models import AutoModel

model = AutoModel.load_from_hf_hub("asenella/test")


# data = set_inputs_to_device(eval_data[:100], device="cuda")
# nll = model.compute_joint_nll(data)
# print(nll)