from config2 import *

from multivae.models import JNFGMC, JNFGMCConfig, GMC, GMCConfig
from multivae.trainers import TwoStepsTrainer, TwoStepsTrainerConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from multivae.models.base import BaseEncoder, ModelOutput
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed)
)

# Define the GMC module

gmc_config = GMCConfig(
    n_modalities=5,
    input_dims=base_config['input_dims'],
    common_dim=64,
    latent_dim=10,
    temperature = 0.1
)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MHDCommonEncoder(BaseEncoder):

    def __init__(self, common_dim, latent_dim):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return ModelOutput(embedding = F.normalize(self.feature_extractor(x), dim=-1))


processors = { k: EncoderConvMMNIST_adapted(BaseAEConfig(latent_dim=gmc_config.common_dim, 
                                                            style_dim=0, input_dim=(3, 28, 28))) 
                 for k in modalities}

gmc_model = GMC(
    model_config=gmc_config,
    processors= processors,
    joint_encoder=MultipleHeadJointEncoder(processors,BaseAEConfig(latent_dim=gmc_config.common_dim)),
    shared_encoder=MHDCommonEncoder(gmc_config.common_dim, gmc_config.latent_dim)
)




model_config = JNFGMCConfig(
    **base_config,
    latent_dim=128,
    warmup=100,
    nb_epochs_gmc=100,
    apply_rescaling=False,
    beta=1.,
)

head_encoders = {
    k: Enc(ndim_w = 0,ndim_u=model_config.latent_dim)
    for k in modalities
}

joint_encoder = MultipleHeadJointEncoder(head_encoders, model_config)



decoders = {m : Dec(ndim=model_config.latent_dim) for m in modalities}



# encoders are default MLP 
model = JNFGMC(
    model_config=model_config,
    gmc_model=gmc_model,
    decoders=decoders,
    joint_encoder=joint_encoder
)

id = [model.model_name,f'seed_{args.seed}']

trainer_config = TwoStepsTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, *id),
    start_keep_best_epoch=0
)
trainer_config.per_device_train_batch_size = 64
trainer_config.per_device_eval_batch_size = 64
trainer_config.num_epochs = model_config.nb_epochs_gmc + model_config.warmup + 100

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = TwoStepsTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model

save_to_hf(model, id)

##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)