from config2 import *
import numpy as np

from multivae.models import MMVAEPlus, MMVAEPlusConfig

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train",
    missing_ratio=args.missing_ratio,
    keep_incomplete=args.keep_incomplete,
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = MMVAEPlusConfig(
    **base_config,
    K=10,
    prior_and_posterior_dist='normal',
    learn_shared_prior=False,
    learn_modality_prior=True,
    beta=1,
    modalities_specific_dim=102,
    reconstruction_option="joint_prior",
)
model_config.latent_dim = 104

##### Architectures #####
from multivae.models.nn.mmnist import EncoderConvMMNIST_multilatents,DecoderConvMMNIST
from multivae.models.base import BaseAEConfig

encoders = {m : EncoderConvMMNIST_multilatents(BaseAEConfig(latent_dim=model_config.latent_dim, 
                                                            style_dim=model_config.modalities_specific_dim,
                                                            input_dim=(3,28,28))) for m in modalities}

decoders = {m : DecoderConvMMNIST(BaseAEConfig(input_dim=(3,28,28), latent_dim=model_config.latent_dim+
                                               model_config.modalities_specific_dim)) for m in modalities}

model = MMVAEPlus(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir=f"compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/K_{model.K}",
)
trainer_config.per_device_train_batch_size = 32
trainer_config.num_epochs = 70 if model.K==1 else 50 # enough for this model to reach convergence

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model
save_model(model, args)
##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir, test_data, wandb_cb.run.path)
