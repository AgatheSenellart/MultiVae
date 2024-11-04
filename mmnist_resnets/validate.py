

from global_config import *
from multivae.models import AutoModel
from multivae.data.datasets import MMNISTDataset
import time
from multivae.trainers.base.callbacks import load_wandb_path_from_folder
import os

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 0


liste_models = [
    '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_3/JNF_training_2024-10-25_23-32-10/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_0/JNF_training_2024-10-25_17-14-46/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_1/JNF_training_2024-10-25_17-22-43/final_model',
    '/home/asenella/scratch/experiments/mmnist_resnets/JNF/seed_2/JNF_training_2024-10-25_17-31-07/final_model',
    
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


for path in liste_models :
    dir_ = path
    
    if os.path.exists(dir_):
            
        # dir = '/home/asenella/experiments/mmnist_resnets/MoPoE/seed_0/MoPoE_training_2024-09-20_17-33-05/final_model'
        wandb_path = load_wandb_path_from_folder(dir_)

        model = AutoModel.load_from_folder(dir_)
        model = model.eval().to(device)
        model.device = device

        
        t1 = time.time()
        eval_model(model, dir_,train_data=train_data, test_data=test_data,wandb_path=wandb_path,seed=seed)
        t2 = time.time()
        print(t2-t1)