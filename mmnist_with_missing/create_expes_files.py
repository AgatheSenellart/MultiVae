"""This file is used to create experiment configurations for the partial data analysis on MMNIST"""

import itertools
import json
from pathlib import Path

Path('config').mkdir( parents=True, exist_ok=True)
Path('config_only_incomplete').mkdir(parents=True, exist_ok=True)

pairs = zip(['config', 'config_only_incomplete'], [[False, True], [False]])

for tup in pairs:
    folder = tup[0]
    
    params_options = {
        "seed": [0, 1, 2, 3],
        "keep_incomplete": tup[1],
        "missing_ratio": [0, 0.2, 0.5],
        
    }

    hypnames, hypvalues = zip(*params_options.items())
    trial_hyperparameter_set = [
        dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)
    ]

    file_nb = 1
    for i, config in enumerate(trial_hyperparameter_set):
        # remove unwanted configurations
        if config["missing_ratio"] == 0 and config["keep_incomplete"]:
            pass
        else:
            with open(f"{folder}/f{file_nb}.json", "w+") as fp:
                json.dump(config, fp)
            file_nb += 1
        
        
