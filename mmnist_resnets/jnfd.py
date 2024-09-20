from config2 import *

from multivae.models import JNFDcca, JNFDccaConfig
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted


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

model_config = JNFDccaConfig(
    **base_config,
    latent_dim=128,
    warmup=100,
    nb_epochs_dcca=100,
    use_all_singular_values=False,
    apply_rescaling=False,
    embedding_dcca_dim=10,
    beta=1.
)

head_encoders = {
    k: Enc(ndim_w = 0,ndim_u=model_config.latent_dim)
    for k in modalities
}

joint_encoder = MultipleHeadJointEncoder(head_encoders, model_config)

# Convolutional networks for the DCCA
dcca_networks = {
    k: EncoderConvMMNIST_adapted(BaseAEConfig(
            latent_dim=model_config.embedding_dcca_dim, style_dim=0, input_dim=(3, 28, 28)
        ))
    for k in modalities
}

    # dcca_networks = {
    #     k: Enc(ndim_w = 0,ndim_u=model_config.embedding_dcca_dim)
    #     for k in modalities
    # }
    

decoders = {m : Dec(ndim=model_config.latent_dim) for m in modalities}



# encoders are default MLP 
model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders, joint_encoder=joint_encoder)

id = [model.model_name,f'seed_{args.seed}']

trainer_config = AddDccaTrainerConfig(
    **base_training_config,
    learning_rate_dcca=1e-4,
    per_device_dcca_train_batch_size=500,
    per_device_dcca_eval_batch_size=500,
    seed=args.seed,
    output_dir=os.path.join(project_path, model.model_name, *id),
)
trainer_config.num_epochs = model_config.nb_epochs_dcca + model_config.warmup + 100

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(args)

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

save_to_hf(model, id)

##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)
