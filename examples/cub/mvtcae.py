import torch
from pythae.models.base.base_config import BaseAEConfig
from torch.utils.data import random_split

from multivae.data.datasets.cub import CUB
from multivae.models import MVTCAE, MVTCAEConfig

from multivae.models.nn.cub import (
   CUB_Resnet_Encoder,
   CUB_Resnet_Decoder,
   CubTextDecoderMLP, 
   CubTextEncoder
)
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

######################################################
### Encoders & Decoders

data_path = '/scratch/asenella/data'

train_data = CUB(
    data_path, "train", captions_per_image=10, im_size=(64, 64), output_type='tokens'
)
eval_data = CUB(data_path, "eval", captions_per_image=10, im_size=(64, 64), output_type='tokens')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = MVTCAEConfig(
    n_modalities=2,
    input_dims={
        "image": (3, 64, 64),
        "text": (train_data.max_words_in_caption, train_data.vocab_size),
    },
    latent_dim=64,
    decoders_dist={"image": "laplace", "text": "categorical"},
    beta=5.0,
    alpha=0.9
)

encoders = {
    "image": CUB_Resnet_Encoder(latent_dim=model_config.latent_dim
    ).to(device),
    "text": CubTextEncoder(
        latent_dim=model_config.latent_dim,
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
    "image": CUB_Resnet_Decoder(latent_dim=model_config.latent_dim
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
    num_epochs=200,
    learning_rate=1e-3,
    steps_predict=5,
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
