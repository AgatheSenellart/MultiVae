"""Script to compute more clustering metrics on the MMNIST experiments"""

import os

from multivae.models import AutoModel
from multivae.metrics import Clustering, ClusteringConfig


from global_config import *

# parse arguments for the experiments
parser = argument_parser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()

# get the hf repo from arguments
hugging_face_path = get_hf_path_from_arguments(args)
model = AutoModel.load_from_hf_hub(hugging_face_path, allow_pickle=True)

# get the datasets
train_data, eval_data, test_data = get_datasets()

output_dir = os.path.join(model_save_path(model, args), 'metrics')


clustering_config = ClusteringConfig(
                number_of_runs=4 # average accuracy on 4 runs
                     )
clustering_module = Clustering(
    model = model,
    test_dataset=test_data,
    train_dataset=train_data,
    eval_config=clustering_config,
    output=output_dir
    )
clustering_module.eval() # might take some time
