from multivae.models import JNFConfig, JNF
from architectures import *
from multivae.models.base import BaseAEConfig
import argparse
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)
import json
from compute_mfd import compute_mfd
import os

wandb_project = 'MHD'
config_name = 'mhd_config_1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
project_path = '/home/asenella/scratch/mhd_experiments/'
data_path = '/home/asenella/scratch/data/MHD'

# project_path = '/Users/agathe/experiments/MHD/'
# data_path = '/Users/agathe/dev/data/MHD'


classifiers_path = os.path.join(data_path, 'classifiers')



base_config = dict(
    n_modalities=3,
    input_dims=dict(image = (3,28,28),
                    audio = (1,32,128),
                    trajectory = (200,))
    )

base_trainer_config = dict(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_epochs=300,
    learning_rate = 1e-3,
    steps_predict=5

)

from multivae.data.datasets.mhd import MHD
from torch.utils.data import random_split
import os

train_set = MHD(data_path, split='train', modalities=['audio', 'trajectory', 'image'])
test_set = MHD(data_path, split='test', modalities=['audio', 'trajectory', 'image'])



classifiers = dict(
    image = Image_Classifier(),
    audio = Sound_Classifier(),
    trajectory = Trajectory_Classifier()
    
)


state_dicts = dict(
    image = torch.load(os.path.join(classifiers_path, 'best_image_classifier_model.pth.tar'), map_location=device)['state_dict'],
    audio = torch.load(os.path.join(classifiers_path, 'best_sound_classifier_model.pth.tar'), map_location=device)['state_dict'],
    trajectory = torch.load(os.path.join(classifiers_path, 'best_trajectory_classifier_model.pth.tar'), map_location=device)['state_dict'],


)

for s in state_dicts:
    classifiers[s].load_state_dict(state_dicts[s])
    classifiers[s].eval()
    

from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, Reconstruction, ReconstructionConfig

def eval(path,model, classifiers, wandb_path):
    
    coherence_config = CoherenceEvaluatorConfig(128, wandb_path=wandb_path)
    CoherenceEvaluator(
        model=model,
        classifiers=classifiers,
        test_dataset=test_set,
        output=path,
        eval_config=coherence_config
        ).eval()
    
    
    for m in ['SSIM', 'MSE']:
                
        recon_config = ReconstructionConfig(
            batch_size=64,
            wandb_path=wandb_path,
            metric=m
        )
        
        recon_module = Reconstruction(
            model, 
            test_dataset=test_set,
            output=path,
            eval_config=recon_config
        )
        
        recon_module.reconstruction_from_subset(['audio'])
        recon_module.reconstruction_from_subset(['image'])
        recon_module.log_to_wandb()
        recon_module.finish()
        
    compute_mfd(model, wandb_path,path)
    
