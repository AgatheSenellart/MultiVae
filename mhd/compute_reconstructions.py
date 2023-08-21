from multivae.models import AutoModel
import os, glob
import json
from config import *
from multivae.metrics.reconstruction import Reconstruction, ReconstructionConfig
from multivae.metrics import Visualization, VisualizationConfig
import wandb

models_name = ['MMVAEPlus', 'MMVAE', 'MVTCAE','MoPoE']
betas = ['beta_5']
rescales = ['True']


for model_name in models_name:
    for beta in betas:
        for rescale in rescales:
  
            for seed in range(4):
                model_path = f'asenella/incomplete_mhd_{model_name}_{beta}_scale_{rescale}_seed_{seed}'
                model = AutoModel.load_from_hf_hub(model_path, allow_pickle=True)
                

                    
                run = wandb.init(entity='multimodal_vaes',
                                        project='validate_mhd',
                                        config=model.model_config.to_dict(),
                                        reinit=True
                                        )
                run.config.update(dict(incomplete = True))
                
                ### Evaluate
                
                for m in ['SSIM', 'MSE']:
                
                    recon_config = ReconstructionConfig(
                        batch_size=64,
                        wandb_path=run.path,
                        metric=m
                    )
                    
                    recon_module = Reconstruction(
                        model, 
                        test_dataset=test_set,
                        # output=model_path,
                        eval_config=recon_config
                    )
                    
                    recon_module.reconstruction_from_subset(['audio'])
                    recon_module.reconstruction_from_subset(['image'])
                    recon_module.log_to_wandb()
                    recon_module.finish()
                
                if seed==0:
                    vis_config = VisualizationConfig(
                        batch_size=128,
                        wandb_path=run.path,
                        n_samples=4,
                        n_data_cond=4
                    )
                    
                    vis_module = Visualization(
                        model,
                        test_set,
                        # output=path,
                        eval_config=vis_config
                    )

                    
                    vis_module.conditional_samples_subset(['audio'])
                    vis_module.conditional_samples_subset(['trajectory'])
                    vis_module.conditional_samples_subset(['image'])
                    
                    vis_module.conditional_samples_subset(['audio', 'trajectory'])
                    vis_module.log_to_wandb()
                    vis_module.finish()



                
                
                
            
            
                