"""This file is used to create experiment configurations for the model comparisons on MMNIST resnets """

import itertools
import json
from pathlib import Path

folder = f'/home/asenella/dev/multivae_package/expes/cub/config_files/JNFGMC'

params_options = {
    "seed": [0],
    "beta": [ 1.0 ], 
    "loss":['between_modality_pairs','between_modality_joint'],
    "annealing" : [True, False],
    "warmup" : [50,100,150],
    "alpha" : [0.1, 0.5]
    
    
}

hypnames, hypvalues = zip(*params_options.items())
trial_hyperparameter_set = [
    dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
]

file_nb = 1
for i, config in enumerate(trial_hyperparameter_set):
    
    if config['annealing'] == False and config['warmup'] != 150:
        continue
    
    if config['annealing'] == False and config['alpha'] != 0.1:
        continue
    
    if config['annealing'] and config["warmup"] == 150:
        continue
    
    with open(f"{folder}/f{file_nb}.json", "w+") as fp:
            json.dump(config, fp)
    file_nb += 1
    
    