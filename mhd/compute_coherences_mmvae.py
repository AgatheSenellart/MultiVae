from multivae.models import AutoModel
import os
import json
from config import *

models_name = ['MMVAEPlus']
betas = ['beta_5']
rescales = ['True']


for model_name in models_name:
    for beta in betas:
        for rescale in rescales:
                        
            
            path = f'/home/asenella/scratch/mhd_experiments/{model_name}/{beta}/rescale_{rescale}'

            list_dirs = os.listdir(path)
            
            
            for m in list_dirs:
                if m != 'metrics.log':
                    model_path = os.path.join(path,m,'final_model')
                    
                    model = AutoModel.load_from_folder(model_path)
                    
                    with open(os.path.join(model_path,'wandb_info.json'),'r') as f:
                        d = json.load(f)
                        
                    eval(model_path, model,classifiers,d['path'])
                
                
                    