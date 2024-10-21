from global_config import *
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
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed)
)

model_config = MMVAEPlusConfig(
    **base_config,
    K=args.K,
    prior_and_posterior_dist='laplace_with_softmax',
    learn_shared_prior=False,
    learn_modality_prior=True,
    beta=args.beta,
    latent_dim=32,
    modalities_specific_dim=32,
    reconstruction_option="joint_prior",
)


encoders = {
    m: Enc(ndim_w = model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
    for m in modalities
}
decoders = {
    m: Dec(ndim = model_config.latent_dim + model_config.modalities_specific_dim)
    for m in modalities
}

model = MMVAEPlus(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    **base_training_config,
    seed=args.seed,
    output_dir= f"~/experiments/mmnist_resnets/{model.model_name}/seed_{args.seed}/",
    optimizer_params=dict(amsgrad=True),

)
trainer_config.per_device_train_batch_size = 32
trainer_config.per_device_eval_batch_size = 32

trainer_config.num_epochs = 150 if model.K==1 else 50 # enough for this model to reach convergence



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


##################################################################################################################################
# validate the model #############################################################################################################
##################################################################################################################################

eval_model(model, trainer.training_dir,train_data, test_data, wandb_cb.run.path,args.seed)

save_to_hf(model, wandb_cb)
