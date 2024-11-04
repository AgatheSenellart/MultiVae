from multivae.models import AutoModel
from multivae.trainers.base.callbacks import load_wandb_path_from_folder

liste_models = [
    # '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_3/JNF_training_2024-10-25_23-32-10/final_model',
    # '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_0/JNF_training_2024-10-25_17-14-46/final_model',
    # '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_1/JNF_training_2024-10-25_17-22-43/final_model',
    # '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_2/JNF_training_2024-10-25_17-31-07/final_model',
    
    '/home/asenella/scratch/experiments/mmnist_resnets/JNFGMC/seed_3/JNFGMC_training_2024-10-26_06-21-41/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNFGMC/seed_0/JNFGMC_training_2024-10-25_23-40-18/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNFGMC/seed_1/JNFGMC_training_2024-10-25_23-48-48/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNFGMC/seed_2/JNFGMC_training_2024-10-26_05-47-47/final_model',
    
    '/home/asenella/scratch/experiments/mmnist_resnets/MoPoE/seed_3/MoPoE_training_2024-10-27_05-50-57/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MoPoE/seed_0/MoPoE_training_2024-10-26_20-55-41/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MoPoE/seed_1/MoPoE_training_2024-10-26_21-33-03/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MoPoE/seed_2/MoPoE_training_2024-10-27_01-59-13/final_model',
    
    '/home/asenella/scratch/experiments/mmnist_resnets/MVTCAE/seed_3/MVTCAE_training_2024-10-26_15-01-48/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MVTCAE/seed_0/MVTCAE_training_2024-10-26_06-32-27/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MVTCAE/seed_1/MVTCAE_training_2024-10-26_12-28-46/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MVTCAE/seed_2/MVTCAE_training_2024-10-26_13-03-51/final_model',
    
    '/home/asenella/scratch/experiments/mmnist_resnets/MMVAE/seed_0/MMVAE_training_2024-10-27_06-32-16/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MMVAE/seed_1/MMVAE_training_2024-10-27_10-56-03/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/MMVAE/seed_2/MMVAE_training_2024-10-27_15-45-03/final_model'
]

for path in liste_models:
    
    model = AutoModel.load_from_folder(path)
    wandb_path = load_wandb_path_from_folder(path)
    wandb_id = wandb_path.split('/')[-1]
    
    model.push_to_hf_hub(f'asenella/mmnist_resnets_{model.model_name}_{wandb_id}')
    
