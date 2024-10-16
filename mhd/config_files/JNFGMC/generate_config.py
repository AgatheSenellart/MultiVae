"""This file is used to create experiment configurations for the model comparisons on MMNIST resnets """

import itertools
import json
from pathlib import Path

# folder = f'/home/asenella/dev/multivae_package/expes/MHD/config_files/JNFGMC'
folder = f'/Users/agathe/dev/multivae_package/expes/MHD/config_files/JNFGMC'


params_options = {
    "seed": [0],
    "beta": [ 0.5 ], 
    "gmc_loss":['between_modality_pairs'],
    "annealing" : [False, True],
    "warmup" : [50,100],
    "alpha" : [0.1],
    "use_rescaling" : [True],
    "temperature" : [0.07,0.1,0.2]
    
    
}

hypnames, hypvalues = zip(*params_options.items())
trial_hyperparameter_set = [
    dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
]

file_nb = 1
for i, config in enumerate(trial_hyperparameter_set):
    
    if config['annealing'] == False and config['warmup'] != 100:
        continue
    
    if config['annealing'] == False and config['alpha'] != 0.1:
        continue
    
    
    with open(f"{folder}/f{file_nb}.json", "w+") as fp:
            json.dump(config, fp)
    file_nb += 1
    
    