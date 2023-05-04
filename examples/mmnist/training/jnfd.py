from config2 import *

from multivae.models import JNFDcca, JNFDccaConfig
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=8)
parser.add_argument('--missing_ratio',type=float, default=0)
args = parser.parse_args()

train_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="train", missing_ratio=args.missing_ratio)
test_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = JNFDccaConfig(
    **base_config,
    warmup=base_training_config["num_epochs"] // 2,
    nb_epochs_dcca=200,
    embedding_dcca_dim=20,
)

dcca_networks = {
    k: encoder_class(
        BaseAEConfig(latent_dim=model_config.embedding_dcca_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = JNFDcca(model_config, dcca_networks=dcca_networks, decoders=decoders)

trainer_config = AddDccaTrainerConfig(
    **base_training_config,
    learning_rate_dcca=1e-4,
    per_device_dcca_train_batch_size=500,
    per_device_dcca_eval_batch_size=500,
    seed=args.seed,
    output_dir= f'compare_on_mmnist/{config_name}/{model.model_name}/seed_{args.seed}/missing_ratio_{args.missing_ratio}/'

)
trainer_config.num_epochs += (
    model_config.nb_epochs_dcca
)  # Add the DCCA time to overall number of epochs

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name=wandb_project)
wandb_cb.run.config.update(dict(missing_ratio=args.missing_ratio))

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = AddDccaTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

# validate the model and save

model = trainer._best_model
coherences = CoherenceEvaluator(
    model=model,
    test_dataset=test_data,
    classifiers=load_mmnist_classifiers(device=model.device),
    output=trainer.training_dir,
).eval()

trainer._best_model.push_to_hf_hub("asenella/mmnist" + model.model_name + config_name)

eval_model(model, trainer.training_dir,test_data,wandb_cb.run.path)
