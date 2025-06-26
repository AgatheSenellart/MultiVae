from multivae.models import MHVAE, MHVAEConfig
from multivae.data.datasets import MnistContourLabels
from architectures import FirstTopDown, Encoder, LastBottomUp,prior_block, posterior_block, Decoder, BottomUpBlock, TopDown

# Paths variables
DATA_PATH = '/Users/agathe/dev/data'


# Define the dataset
train_dataset = MnistContourLabels(DATA_PATH, split='train',random_binarized=False, include_labels=False, include_contours=True)
test_dataset = MnistContourLabels(DATA_PATH, split='test', random_binarized=False, include_labels=False, include_contours=True)

# Define the model
model_config = MHVAEConfig(
    n_modalities=2,
    latent_dim=20,
    input_dims={'images':(1,28,28), 'contours':(1,28,28)},
    n_latent=4
)

# Define the architectures
encoders = {'images':Encoder(4), 'contours':Encoder(4)} # (4,28,28)
bottom_ups = {
    'images': [BottomUpBlock(4,8), # (8,14,14)
               BottomUpBlock(8,16), # (16, 7,7)
               LastBottomUp(16,32,32*4*4,model_config.latent_dim) # latent_dim
               ],
    'contours': [BottomUpBlock(4,8), # (8,14,14)
               BottomUpBlock(8,16), # (16, 7,7)
               LastBottomUp(16,32,32*4*4,model_config.latent_dim) # latent_dim
               ]
}
top_downs = [ FirstTopDown(model_config.latent_dim,(32,4,4), 32, 16, output_size=(7,7)),
               TopDown(16,8,(14,14)),
               TopDown(8,4,(28,28))
    ] 

top_downs.reverse()

priors = [
    prior_block(16), 
    prior_block(8),
    prior_block(4)
]

priors.reverse()

posteriors = {
    'images': [posterior_block(4), posterior_block(8), posterior_block(16)],
    'contours': [posterior_block(4), posterior_block(8), posterior_block(16)]
}

decoders = {
    'images' : Decoder(n_channels=4, nb_of_blocks=3),
    'contours': Decoder(n_channels=4, nb_of_blocks=3)
}

# Finally define the model
model = MHVAE(
    model_config=model_config,
    encoders=encoders,
    decoders=decoders,
    bottom_up_blocks=bottom_ups, 
    top_down_blocks=top_downs, 
    posterior_blocks=posteriors,
    prior_blocks=priors
)


from multivae.trainers import BaseTrainer, BaseTrainerConfig

trainer_config = BaseTrainerConfig(
    learning_rate=1e-3
)

trainer = BaseTrainer(
    model, train_dataset,None, trainer_config
)

trainer.train()