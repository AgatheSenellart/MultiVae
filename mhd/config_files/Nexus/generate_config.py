"""This file is used to create experiment configurations for the model comparisons on MMNIST resnets """

import itertools
import json
from pathlib import Path

folder = f'/home/asenella/dev/multivae_package/expes/mhd/config_files/Nexus'
# folder = f'/Users/agathe/dev/multivae_package/expes/MHD/config_files/Nexus'


params_options = {
    "seed": [0],
    "top_beta": [0.5,1.0,2.5], 
    "use_rescaling" : [True],
    "latent_dim" : [64]
    
    
    
}

hypnames, hypvalues = zip(*params_options.items())
trial_hyperparameter_set = [
    dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
]

file_nb = 1
for i, config in enumerate(trial_hyperparameter_set):

    with open(f"{folder}/f{file_nb}.json", "w+") as fp:
            json.dump(config, fp)
    file_nb += 1
    
    