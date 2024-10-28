from multivae.models import AutoModel
from multivae.trainers.base.callbacks import load_wandb_path_from_folder

liste_models = [
    'path'
]

for path in liste_models:
    
    model = AutoModel.load_from_folder(path)
    wandb_path = load_wandb_path_from_folder(path)
    wandb_id = wandb_path.split('/')[-1]
    
    model.push_to_hf_hub(f'asenella/mmnist_resnets_{model.model_name}_{wandb_id}')
    
