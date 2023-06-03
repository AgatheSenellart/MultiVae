import torch
from pythae.models.base.base_config import BaseAEConfig
from multivae.data.datasets.cub import CUB
from multivae.models import MVTCAE, MVTCAEConfig
from multivae.models.nn.default_architectures import (
    BaseDecoder,
    BaseEncoder,
    ModelOutput,
)
from torch.utils.data import random_split
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback
from multivae.models.nn.cub import CubTextEncoder, CubTextDecoderMLP
from multivae.models.nn.mmnist import Decoder_ResNet_AE_MMNIST, Encoder_ResNet_VAE_MMNIST
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
######################################################
### Encoders & Decoders

cub = CUB('../../data/CUB/CUB_200_2011/', 'test', captions_per_image=10, im_size=(28, 28))
train_data, eval_data = random_split(
    cub, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)
print(len(train_data), len(eval_data))

model_config = MVTCAEConfig(
    n_modalities=2,
    input_dims={
        'img': (3, 28, 28),
        'text': (cub.max_words_in_captions, cub.vocab_size)
    },
    latent_dim=16,
    decoders_dist={
        "img": "laplace",
        "text": "categorical"},
    beta=2.5,
    alpha=5.0 / 6.0,
)

encoders = {
    'img': Encoder_ResNet_VAE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    ).cuda(),
    'text': CubTextEncoder(
        model_config,
        max_sentence_length=cub.max_words_in_captions,
        ntokens=cub.vocab_size,
        embed_size=512,
        ff_size=128,
        n_layers=2,
        nhead=2,
        dropout=0.1
    ).cuda()
}

decoders = {
    'img': Decoder_ResNet_AE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    ).cuda(),
    'text': CubTextDecoderMLP(
    BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(cub.max_words_in_captions, cub.vocab_size))
).cuda()
    
}

model = MVTCAE(model_config, encoders=encoders, decoders=decoders).cuda()

trainer_config = BaseTrainerConfig(
    num_epochs=1000,
    learning_rate=1e-3,
    steps_predict=None,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
)

## Set up callbacks
#wandb_cb = WandbCallback()
#wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

callbacks = [TrainingCallback()]#, wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    #eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()