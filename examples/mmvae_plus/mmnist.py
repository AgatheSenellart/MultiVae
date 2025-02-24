"""In this file, we reproduce the MMVAE+ results on the PolyMNIST dataset. """

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.trainers.base.callbacks import  WandbCallback
from multivae.models.mmvaePlus import MMVAEPlus, MMVAEPlusConfig
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig
from multivae.metrics.coherences import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.metrics.fids import FIDEvaluator, FIDEvaluatorConfig

from architectures import Enc, Dec, load_mmnist_classifiers

# Define paths 
DATA_PATH = '/home/asenella/data'
SAVE_PATH = '/home/asenella/experiments'


# Define model
modalities = ["m0", "m1", "m2", "m3", "m4"]
model_config = MMVAEPlusConfig(
    n_modalities=5,
    K=1,
    decoders_dist={m: "laplace" for m in modalities},
    decoder_dist_params={m: dict(scale=0.75) for m in modalities},
    prior_and_posterior_dist="laplace_with_softmax",
    beta=2.5,
    modalities_specific_dim=32,
    latent_dim=32,
    input_dims={m: (3, 28, 28) for m in modalities},
    learn_shared_prior=False,
    learn_modality_prior=True,
    loss='iwae_looser'
)

encoders = {
    m: Enc(model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
    for m in modalities
}
decoders = {
    m: Dec(model_config.latent_dim + model_config.modalities_specific_dim)
    for m in modalities
}

model = MMVAEPlus(model_config, encoders, decoders)


######## Dataset #########

train_data = MMNISTDataset(data_path=DATA_PATH, split="train")
test_data = MMNISTDataset(data_path=DATA_PATH, split="test")


########## Training #######

training_config = BaseTrainerConfig(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_epochs=50 if model_config.K == 10 else 150,
    learning_rate=1e-3,
    output_dir=f"{SAVE_PATH}/reproduce_mmvaePlus/K__{model_config.K}",
    steps_predict=5,
    optimizer_cls="Adam",
    optimizer_params=dict(amsgrad=True),
    seed=0,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae_plus")

callbacks = [wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=None,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

#### Validation ####

# Compute Coherences
config = CoherenceEvaluatorConfig(batch_size=512, 
                                  wandb_path=wandb_cb.run.path
                                  )

CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(data_path=DATA_PATH + '/clf', device=model.device),
    output=trainer.training_dir,
    eval_config=config,
).eval()

# Compute FID
config = FIDEvaluatorConfig(batch_size=512, 
                            wandb_path=wandb_cb.run.path, 
                            inception_weights_path=DATA_PATH + '/pt_inception-2015-12-05-6726825d.pth'
                            )

fid = FIDEvaluator(
    model, test_data, output=trainer.training_dir, eval_config=config
).eval()
