from multivae.models import AutoModel
import os, glob
import json
from config import *
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
from multivae.trainers.base.callbacks import load_wandb_path_from_folder
import wandb
from multivae.data.datasets import MultimodalBaseDataset
import torch
from pythae.models import AutoModel as pythae_automodel

models_name = [ 'MVTCAE', 'MoPoE', 'JNF', 'MMVAE', 'MMVAEPlus']
betas = ['beta_5']
rescales = ['True']


def compute_mfd(model, wandb_path, path):
    
    # import the training data by hand to filter it by label.
    data = dict()

    (
                data['label'],
                data['image'],
                data['trajectory'],
                data['audio'],
                _traj_normalization,
                _audio_normalization,
            ) = torch.load('/home/asenella/scratch/data/MHD/mhd_test.pt')



    def unstack_tensor(tensor, dim=0):
        tensor_lst = []
        for i in range(tensor.size(dim)):
            tensor_lst.append(tensor[i])
        tensor_unstack = torch.cat(tensor_lst, dim=0)
        return tensor_unstack

    _a_data = data['audio'].permute(1,2,3,0)
    _a_data = unstack_tensor(_a_data,dim=0).unsqueeze(0)
    data['audio'] = _a_data.permute(3,0, 2,1)

    for label in range(10):
                    
        # get only the samples corresponding to the label
        dataset = MultimodalBaseDataset(
            data = {m : data[m][data['label']==label] for m in ['audio', 'trajectory', 'image']}, 
            labels=data['label'][data['label']==label]
        )
        
        

        # get the encoders
        
        encoders = {
            m: pythae_automodel.load_from_folder(
                    os.path.join(f'/home/asenella/scratch/mhd_unimodal_encoders/{m}/{label}',
                        os.listdir(f'/home/asenella/scratch/mhd_unimodal_encoders/{m}/{label}')[0],
                        'final_model')
            ).encoder for m in ['audio', 'trajectory', 'image']
        }
        
        # Configure FID

        fid_config = FIDEvaluatorConfig(
            batch_size=128,
            wandb_path=wandb_path
            
        )
        
        fid_module = FIDEvaluator(
            model = model,
            test_dataset=dataset,
            output=path,
            eval_config=fid_config,
            custom_encoders=encoders
        )

        
        mfd = 0
        
        for mod in encoders:
            for gen_mod in encoders:
                if gen_mod != mod:
                    mfd += fid_module.compute_fid_from_conditional_generation([mod], gen_mod)
        
        fid_module.metrics[f'MFD_label_{label}'] = mfd 
        fid_module.log_to_wandb()
        fid_module.finish()



if __name__ == '__main__':
    for model_path in ['/home/asenella/scratch/mhd_experiments/JNFDcca/beta_5/rescale_True/JNFDcca_training_2024-10-19_04-28-55/final_model',
                       '/home/asenella/scratch/mhd_experiments/JNFGMC/beta_5/rescale_True/JNFGMC_training_2024-10-18_20-26-09/final-model']:
        model = AutoModel.load_from_folder(model_path)
        wandb_path = load_wandb_path_from_folder(model_path)
        compute_mfd(model,wandb_path, None)