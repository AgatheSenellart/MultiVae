from multivae.models import JMVAE, JMVAEConfig
from multivae.data.datasets import MnistSvhn
from pythae.models.nn.default_architectures import Encoder_VAE_MLP, Decoder_AE_MLP
from pythae.models.base.base_config import BaseAEConfig
from multivae.models.nn.svhn import Encoder_VAE_SVHN, Decoder_VAE_SVHN
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from torch.utils.data import DataLoader
from multivae.data.datasets.utils import save_all_images
from torch.utils.data import random_split
import torch

train_data = MnistSvhn(split='test')
train_data, eval_data = random_split(train_data, [0.8,0.2], generator=torch.Generator().manual_seed(42) )

model_config = JMVAEConfig(n_modalities = 2,
                           input_dims=dict(mnist= (1,28,28),
                                           svhn= (3,32,32)),
                           latent_dim=20
                           )

encoders = dict(
    mnist = Encoder_VAE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1,28,28))),
    svhn = Encoder_VAE_SVHN(BaseAEConfig(latent_dim=20,input_dim=(3,32,32)))
)

decoders = dict(
    mnist = Decoder_AE_MLP(BaseAEConfig(latent_dim=20, input_dim=(1,28,28))),
    svhn = Decoder_VAE_SVHN(BaseAEConfig(latent_dim=20,input_dim=(3,32,32)))
)

model = JMVAE(model_config,encoders, decoders)

trainer_config = BaseTrainerConfig(num_epochs=1,learning_rate=1e-3,steps_predict=1)
trainer = BaseTrainer(model,train_dataset=train_data, eval_dataset=eval_data,training_config=trainer_config)
trainer.train()
# save_all_images(model.predict(test_data[:10],'all', 'all'),trainer.training_dir,'_reconstructions')
