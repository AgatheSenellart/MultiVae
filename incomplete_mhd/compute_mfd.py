from multivae.models import AutoModel
import os, glob
import json
from config import *
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig
import wandb
from multivae.data.datasets import MultimodalBaseDataset
import torch
from pythae.models import AutoModel as pythae_automodel

models_name = ['MVTCAE', 'MoPoE', 'MMVAE', 'MMVAEPlus']
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
        
        run.log({f'MFD_label_{label}' : mfd })
        fid_module.log_to_wandb()
        fid_module.finish()




for model_name in models_name:
    for beta in betas:
        for rescale in rescales:

            for seed in range(4):
                
                model_path = f'asenella/incomplete_mhd_{model_name}_{beta}_scale_{rescale}_seed_{seed}'
                model = AutoModel.load_from_hf_hub(model_path, allow_pickle=True)
                
                    
                run = wandb.init(entity='multimodal_vaes',
                                        project='validate_mhd_mfd',
                                        config=model.model_config.to_dict(),
                                        reinit=True
                                        )
                run.config.update(dict(incomplete = True))
                
                compute_mfd(model, run.path, None)