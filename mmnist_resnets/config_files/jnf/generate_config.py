"""This file is used to create experiment configurations for the model comparisons on MMNIST resnets """

import itertools
import json
from pathlib import Path

folder = f'/home/asenella/dev/multivae_package/expes/mmnist_resnets/config_files/jnf'

params_options = {
    "seed": [0,1,2,3],
    "two_steps_training": [True],
    "beta": [2.5], # same choices as in MMVAE+ paper
    "latent_dim" : [190]
    
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
    
    