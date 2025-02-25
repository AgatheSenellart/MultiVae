"""Main code for training a MVTCAE model on the CUB dataset"""

from pythae.models.base.base_config import BaseAEConfig

from multivae.data.datasets.cub import CUB
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.cub import (
    CUB_Resnet_Decoder,
    CUB_Resnet_Encoder,
    CubTextDecoderMLP,
    CubTextEncoder,
)
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback

# Set data path and experiment path
DATA_PATH = "/home/asenella/data"
SAVING_PATH = "/home/asenella/expes/mvtcae_cub"

# Import the dataset
train_data = CUB(
    DATA_PATH, "train", im_size=(64, 64), output_type="tokens", download=True
)
eval_data = CUB(
    DATA_PATH, "eval", im_size=(64, 64), output_type="tokens", download=True
)
# Set up model configuration
model_config = MVTCAEConfig(
    n_modalities=2,
    input_dims={
        "image": (3, 64, 64),
        "text": (train_data.max_words_in_caption, train_data.vocab_size),
    },
    latent_dim=64,
    decoders_dist={"image": "laplace", "text": "categorical"},
    beta=5.0,
    alpha=0.9,
)
# Set up encoders and decoders
encoders = {
    "image": CUB_Resnet_Encoder(latent_dim=model_config.latent_dim),
    "text": CubTextEncoder(
        latent_dim=model_config.latent_dim,
        max_sentence_length=train_data.max_words_in_caption,
        ntokens=train_data.vocab_size,
        embed_size=512,
        ff_size=128,
        n_layers=2,
        nhead=2,
        dropout=0.1,
    ),
}

decoders = {
    "image": CUB_Resnet_Decoder(latent_dim=model_config.latent_dim),
    "text": CubTextDecoderMLP(
        BaseAEConfig(
            latent_dim=model_config.latent_dim,
            input_dim=(train_data.max_words_in_caption, train_data.vocab_size),
        )
    ),
}

# Create the model
model = MVTCAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=200,
    learning_rate=1e-3,
    steps_predict=5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    output_dir=SAVING_PATH,
)

# Set up callbacks and train
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mvtcae_cub")


trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=[wandb_cb],
)
trainer.train()
