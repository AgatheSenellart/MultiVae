from multivae.models import AutoModel
import os
import json

models_name = ['MVTCAE']
betas = ['beta_5','beta_10','beta_25']
rescales = ['True', 'False']
seeds = [0,1,2,3]

for model_name in models_name:
    for seed in seeds:
        for rescale in rescales:
            for beta in betas:

                path = f'/home/asenella/scratch/ms_experiments/{model_name}/{beta}/rescale_{rescale}/seed_{seed}'

                list_dirs = os.listdir(path)
                
                
                for m in list_dirs:
                    if m != 'metrics.log':
                        model_path = os.path.join(path,m,'final_model')
                        
                        model = AutoModel.load_from_folder(model_path)
                    
                        hf_path = f'asenella/ms_{model_name}_{beta}_scale_{rescale}_seed_{seed}'
                        model.push_to_hf_hub(hf_path)