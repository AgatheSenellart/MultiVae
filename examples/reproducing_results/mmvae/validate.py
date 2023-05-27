import torch
from classifiers import load_mnist_svhn_classifiers

from multivae.data.datasets.mnist_svhn import MnistSvhn
from multivae.metrics import (
    CoherenceEvaluator,
    LikelihoodsEvaluator,
    LikelihoodsEvaluatorConfig,
)
from multivae.models import AutoModel

# data_path = "dummy_output_dir/mmvae/final_model"
# model = AutoModel.load_from_folder(data_path)

data_path = None
model = AutoModel.load_from_hf_hub(
    "asenella/reproduce_mmvae_azure_lake", allow_pickle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(model.prior_mean, model.prior_log_var)

clfs = load_mnist_svhn_classifiers("../../classifiers", device=device)

test_set = MnistSvhn(split="test", data_multiplication=30)
print(len(test_set))
# output = CoherenceEvaluator(model, clfs, test_set, data_path).joint_coherence()

lik_config = LikelihoodsEvaluatorConfig(
    batch_size=12, batch_size_k=1000, unified_implementation=False, num_samples=1000
)
output = LikelihoodsEvaluator(
    model, test_set, data_path, eval_config=lik_config
).joint_nll()
# output = CoherenceEvaluator(model,clfs,test_set,data_path).eval()
