from config2 import *

from multivae.models import JNFDcca, JNFDccaConfig
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder, Encoder_VAE_MLP
from multivae.models.nn.mmnist import Encoder_ResNet_VAE_MMNIST


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
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed)
)

model_config = JNFDccaConfig(
    **base_config,
    latent_dim=128,
    warmup=200,
    nb_epochs_dcca=30,
    embedding_dcca_dim=32,
    apply_rescaling=True
)

head_encoders = {
    k: Enc(ndim_w = 0,ndim_u=model_config.latent_dim)
    for k in modalities
}

joint_encoder = MultipleHeadJointEncoder(head_encoders, model_config)

dcca_networks = {
    k: Enc(ndim_w = 0,ndim_u=model_config.embedding_dcca_dim)
        # Encoder_ResNet_VAE_MMNIST(BaseAEConfig(
        #     latent_dim=model_config.embedding_dcca_dim, style_dim=0, input_dim=(3, 28, 28)
        # ))
    for k in modalities
}

decoders = {m : Dec(ndim=model_config.latent_dim) for m in modalities}

encoders = {m : Encoder_VAE_MLP(BaseAEConfig(latent_dim = model_config.latent_dim, 
                                     input_dim = (model_config.embedding_dcca_dim,)), num_hidden=2) for m in modalities}


model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders, joint_encoder=joint_encoder,
                encoders = encoders)

trainer_config = AddDccaTrainerConfig(
    **base_training_config,
    learning_rate_dcca=1e-4,
    per_device_dcca_train_batch_size=800,
    per_device_dcca_eval_batch_size=800,
    seed=args.seed,
    output_dir=f"compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/",
)
trainer_config.num_epochs = 200+200+30

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args.__dict__)

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = AddDccaTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

model = trainer._best_model
save_model(model, args)
#################################################################################################################################
### validate the model #############################################################################################################
#################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)