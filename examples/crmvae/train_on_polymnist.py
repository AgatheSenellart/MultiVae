from multivae.data.datasets import MMNISTDataset
from multivae.models import CRMVAE, CRMVAEConfig
from multivae.models.nn.mmnist import EncoderConvMMNIST, DecoderConvMMNIST
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback
from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, FIDEvaluator, FIDEvaluatorConfig
from multivae.metrics.classifiers.mmnist import load_mmnist_classifiers
from torch.utils.data import random_split

DATA_PATH = '/scratch/asenella/data'
SAVE_PATH = '/scratch/asenella/experiments/CRMVAE_on_MMNIST'
CLASSIFIER_PATH = DATA_PATH + '/clf'
FID_PATH = '/scratch/asenella/fid/pt_inception-2015-12-05-6726825d.pth'

# Download data
train_data = MMNISTDataset(DATA_PATH, download=True)
train_data, eval_data = random_split(train_data,[0.9,0.10])
test_data = MMNISTDataset(DATA_PATH, split='test', download=True)

modalities = ['m0', 'm1', 'm2', 'm3', 'm4']

# Define model config
model_config = CRMVAEConfig(n_modalities=5,
                            latent_dim=512,
                            input_dims={m: (3,28,28) for m in modalities}, 
                            uses_likelihood_rescaling=False,
                            decoders_dist = {m:'laplace' for m in modalities},
                            decoder_dist_params={m:{'scale':0.75} for m in modalities},
                            beta=1.0
                            )

# Define model
model = CRMVAE(model_config=model_config,
               encoders={m : EncoderConvMMNIST(model_config, bias=True) for m in modalities},
               decoders={m: DecoderConvMMNIST(model_config) for m in modalities})


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
wandb_cb.setup(trainer_config,model_config, project_name='crmvae_polymnist')

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

classifiers = load_mmnist_classifiers(CLASSIFIER_PATH)

# Coherence
coherence_config = CoherenceEvaluatorConfig(batch_size=256,wandb_path=wandb_cb.run.path)
coherence_module= CoherenceEvaluator(best_model,classifiers,test_data,output=trainer.training_dir,eval_config=coherence_config)

coherence_module.eval()
coherence_module.finish()

# Fid
fid_config = FIDEvaluatorConfig(batch_size=512,wandb_path=wandb_cb.run.path,inception_weights_path=FID_PATH)
fid_module = FIDEvaluator(best_model,test_data,output=trainer.training_dir)
fid_module.eval()
fid_module.compute_all_conditional_fids(gen_mod='m0')
fid_module.finish()
