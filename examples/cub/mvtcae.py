import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets.cub import CUB
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.cub import CubTextDecoderMLP, CubTextEncoder
from multivae.models.nn.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
)
from multivae.models.nn.mmnist import (
    Decoder_ResNet_AE_MMNIST,
    Encoder_ResNet_VAE_MMNIST,
)
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

######################################################
### Encoders & Decoders

train_data = CUB(
    "/Users/agathe/dev/data", "train", captions_per_image=10, im_size=(28, 28), output_type='tokens'
)
eval_data = CUB("/Users/agathe/dev/data", "eval", captions_per_image=10, im_size=(28, 28), output_type='tokens')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(len(train_data))

model_config = MVTCAEConfig(
    n_modalities=2,
    input_dims={
        "image": (3, 28, 28),
        "text": (train_data.max_words_in_caption, train_data.vocab_size),
    },
    latent_dim=16,
    decoders_dist={"image": "laplace", "text": "categorical"},
    beta=2.5,
    alpha=5.0 / 6.0,
)

encoders = {
    "image": Encoder_ResNet_VAE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    ).to(device),
    "text": CubTextEncoder(
        model_config,
        max_sentence_length=train_data.max_words_in_caption,
        ntokens=train_data.vocab_size,
        embed_size=512,
        ff_size=128,
        n_layers=2,
        nhead=2,
        dropout=0.1,
    ).to(device),
}

decoders = {
    "image": Decoder_ResNet_AE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    ).to(device),
    "text": CubTextDecoderMLP(
        BaseAEConfig(
            latent_dim=model_config.latent_dim,
            input_dim=(train_data.max_words_in_caption, train_data.vocab_size),
        )
    ).to(device),
}

model = MVTCAE(model_config, encoders=encoders, decoders=decoders).to(device)

trainer_config = BaseTrainerConfig(
    num_epochs=1000,
    learning_rate=1e-3,
    steps_predict=None,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    device=device,
)

## Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="test_cub")

callbacks = [TrainingCallback() , wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks
)
trainer.train()
