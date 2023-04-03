import torch
from classifiers import load_mnist_svhn_classifiers
from multivae.data.datasets.mnist_svhn import MnistSvhn
from multivae.metrics import CoherenceEvaluator
from multivae.models import AutoModel


data_path = 'dummy_output_dir/MMVAE_training_2023-04-02_14-57-58/final_model'
model = AutoModel.load_from_folder(data_path)

# model = AutoModel.load_from_hf_hub('asenella/reproducing_mmvae_2',allow_pickle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(model.prior_mean, model.prior_log_var)

clfs = load_mnist_svhn_classifiers('../../classifiers', device=device)

test_set = MnistSvhn(split='test', data_multiplication=30)

output = CoherenceEvaluator(model,clfs,test_set,data_path).eval()