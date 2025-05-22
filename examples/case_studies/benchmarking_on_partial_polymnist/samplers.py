"""Script to compute additional joint generation metrics on the MMNIST experiments"""

import os

from global_config import *
from pythae.trainers import BaseTrainerConfig

from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig
from multivae.models import AutoModel
from multivae.samplers import GaussianMixtureSampler, MAFSampler, MAFSamplerConfig

# Parse arguments for the experiments
parser = argument_parser()
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

# Get the hf repo from arguments
hugging_face_path = get_hf_path_from_arguments(args)
model = AutoModel.load_from_hf_hub(hugging_face_path, allow_pickle=True)

# Get the datasets
train_data, eval_data, test_data = get_datasets()

output_dir = os.path.join(model_save_path(model, args), "metrics")

# Train a MAF Sampler
sampler_training_config = BaseTrainerConfig(
    per_device_train_batch_size=256, num_epochs=20, learning_rate=1e-3
)
sampler_config = MAFSamplerConfig()
maf_sampler = MAFSampler(model)
maf_sampler.fit(
    train_data=train_data, eval_data=eval_data, training_config=sampler_training_config
)

# Train a GMM Sampler
gmm_sampler = GaussianMixtureSampler(model)
gmm_sampler.fit(train_data)

# Compute joint coherence with different samplers
samplers = [maf_sampler, gmm_sampler, None]
classifiers = load_mmnist_classifiers(CLASSIFIER_PATH, device="cpu")

for sampler in samplers:
    config = CoherenceEvaluatorConfig(batch_size=128)
    module_eval = CoherenceEvaluator(
        model, classifiers, test_data, eval_config=config, sampler=sampler
    )
    module_eval.joint_coherence()
    module_eval.finish()

    # Compute joint FID with different samplers
    config = FIDEvaluatorConfig(batch_size=128, inception_weights_path=FID_PATH)
    module_eval = FIDEvaluator(model, test_data, eval_config=config, sampler=sampler)
    module_eval.eval()
    module_eval.finish()
