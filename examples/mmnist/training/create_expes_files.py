'''This file is used to create experiment configurations for the partial data analysis on MMNIST'''

import itertools
import json


params_options = {'missing_ratio': [0, 0.2, 0.5],
                  'keep_incomplete': [True, False],
                  'seed': [0, 1, 2, 3]}

hypnames, hypvalues = zip(*params_options.items())
trial_hyperparameter_set = [dict(zip(hypnames, h)) for h in itertools.product(*hypvalues)]

for i,config in enumerate(trial_hyperparameter_set):
    
    # remove unwanted configurations
    if config['missing_ratio'] == 0 and config['keep_incomplete']:
        pass
    else:
        with open(f'config/f{i}.json', 'a+') as fp:
            json.dump(config,fp)