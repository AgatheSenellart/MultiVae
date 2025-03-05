from multivae.data.datasets import TranslatedMMNIST
from multivae.models import CRMVAE, CRMVAEConfig
from multivae.models.nn.mmnist import EncoderResnetMMNIST, DecoderResnetMMNIST
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, FIDEvaluator, FIDEvaluatorConfig
from torch.utils.data import random_split
from classifiers import load_classifiers


DATA_PATH = '/scratch/asenella/data'
MMNIST_BACKGROUND_PATH = DATA_PATH + '/mmnist_background'
SAVE_PATH = '/scratch/asenella/experiments/CRMVAE_on_MMNIST'
CLASSIFIER_PATH = "/home/asenella/scratch/data/translated_mmnist_2/classifiers"
FID_PATH = DATA_PATH + '/pt_inception-2015-12-05-6726825d.pth'

# Download data
train_data = TranslatedMMNIST(DATA_PATH,scale=0.75,
                              translate=True, 
                              n_modalities=5,background_path=MMNIST_BACKGROUND_PATH,split='train')
train_data, eval_data = random_split(train_data,[0.85,0.15])
test_data = TranslatedMMNIST(DATA_PATH,scale=0.75,
                             translate=True, 
                             n_modalities=5,background_path=MMNIST_BACKGROUND_PATH,split='test')

modalities = ['m0', 'm1', 'm2', 'm3', 'm4']

# Define model config
model_config = CRMVAEConfig(n_modalities=5,
                            latent_dim=512,
                            input_dims={m: (3,28,28) for m in modalities}, 
                            uses_likelihood_rescaling=False,
                            decoders_dist = {m:'laplace' for m in modalities},
                            decoder_dist_params={m:{'scale':0.75} for m in modalities},
                            beta=0.1
                            )

# Define model
model = CRMVAE(model_config=model_config,
               encoders={m : EncoderResnetMMNIST(0,model_config.latent_dim) for m in modalities},
               decoders={m: DecoderResnetMMNIST(model_config.latent_dim) for m in modalities})


# Define training config

trainer_config=BaseTrainerConfig(
    output_dir=SAVE_PATH,
    per_device_train_batch_size= 256,
    per_device_eval_batch_size=256,
    num_epochs=500,
    optimizer_cls='Adam',
    learning_rate=0.0005,
    drop_last=True,
    steps_predict=5 # to visualize generations during training

)

wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config,model_config, project_name='crmvae_tpolymnist')

trainer = BaseTrainer(
    model= model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=[wandb_cb]
)

trainer.train()

# Evaluate for coherence and FID
best_model = trainer._best_model

classifiers = load_classifiers(CLASSIFIER_PATH)

# Coherence
coherence_config = CoherenceEvaluatorConfig(batch_size=256,wandb_path=wandb_cb.run.path)
coherence_module= CoherenceEvaluator(best_model,classifiers,test_data,output=trainer.training_dir,eval_config=coherence_config)

coherence_module.eval()

