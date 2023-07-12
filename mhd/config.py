from multivae.models import JNFConfig, JNF
from architectures import *
from multivae.models.base import BaseAEConfig
import argparse
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

wandb_project = 'MHD'
config_name = 'mhd_config_1'
device = 'cpu'
project_path = './experiments_results'

base_config = dict(
    n_modalities=3,
    latent_dim=64,

)


base_trainer_config = dict(
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_epochs=500,
    learning_rate = 1e-3,
    steps_predict=5

)

from multivae.data.datasets.mhd import MHD
from torch.utils.data import random_split
import os

train_set = MHD('/home/asenella/scratch/data/MHD/mhd_train.pt', modalities=['audio', 'trajectory', 'image'])
test_set = MHD('/home/asenella/scratch/data/MHD/mhd_test.pt', modalities=['audio', 'trajectory', 'image'])


classifiers_path = '/home/asenella/scratch/data/MHD/classifiers'

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
    

from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig, FIDEvaluator, FIDEvaluatorConfig

def eval(path,model, classifiers, wandb_path):
    
    coherence_config = CoherenceEvaluatorConfig(128, wandb_path=wandb_path)
    CoherenceEvaluator(
        model=model,
        classifiers=classifiers,
        test_dataset=test_set,
        output=path,
        eval_config=coherence_config
        ).eval()
    
    
    config = FIDEvaluatorConfig(batch_size=512, wandb_path=wandb_path)

    FIDEvaluator(
        model, test_set, output=path, eval_config=config
    ).compute_all_conditional_fids(gen_mod="image")
    
    