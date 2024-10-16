from multivae.models import AutoModel
import os
import json

models_name = ['JMVAE', 'MoPoE','MVTCAE','JNF']
betas = ['beta_25','beta_10','beta_5']
rescales = ['True', 'False']

if __name__ == '__main__':
    for model_name in models_name:
        for beta in betas:
            for rescale in rescales:

                path = f'/home/asenella/scratch/mhd_experiments/{model_name}/{beta}/rescale_{rescale}'

                list_dirs = os.listdir(path)
                
                
                for m in list_dirs:
                    if m != 'metrics.log':
                        model_path = os.path.join(path,m,'final_model')
                        
                        with open(os.path.join(model_path,'training_config.json'),'r') as f:
                            d = json.load(f)
                            
                        seed = d['seed']
                        
                        model = AutoModel.load_from_folder(model_path)
                    
                        hf_path = f'asenella/{model_name}_{beta}_scale_{rescale}_seed_{seed}'
                        model.push_to_hf_hub(hf_path)