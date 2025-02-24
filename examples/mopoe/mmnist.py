"""In this file, we reproduce the MoPoE results on PolyMNIST"""


from pythae.models.base.base_config import BaseAEConfig

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.models import MoPoE, MoPoEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback

from architectures import EncoderImg, DecoderImg, load_mmnist_classifiers

# Define paths 
DATA_PATH = '/home/asenella/data'
SAVE_PATH = '/home/asenella/experiments'

train_data = MMNISTDataset(data_path=DATA_PATH, split="train")
test_data = MMNISTDataset(data_path=DATA_PATH, split="test")


# Define model configuration
modalities = ["m0", "m1", "m2", "m3", "m4"]
model_config = MoPoEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=512,
    decoders_dist={m: "laplace" for m in modalities},
    decoder_dist_params={m: {"scale": 0.75} for m in modalities},
    beta=2.5,
)


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

model = MoPoE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=300,
    learning_rate=1.0e-3,
    steps_predict=1,
    per_device_train_batch_size=256,
    drop_last=True,
    seed=0,
    output_dir=SAVE_PATH
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="reproducing_mopoe")

callbacks = [wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    training_config=trainer_config,
    callbacks=callbacks,
)

trainer.train()
model = trainer._best_model

#####################################################################################
##### Validation : Compute the coherences

clfs = load_mmnist_classifiers(data_path=DATA_PATH + '/clf')
coherences = CoherenceEvaluator(model, clfs, test_data, trainer.training_dir).eval()

