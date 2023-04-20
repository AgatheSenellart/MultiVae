from classifiers import load_mnist_svhn_classifiers

from multivae.data.datasets.mnist_svhn import MnistSvhn
from multivae.metrics import CoherenceEvaluator
from multivae.models import AutoModel

data_path = "dummy_output_dir/MMVAE_training_2023-03-30_17-29-53/final_model"
model = AutoModel.load_from_folder(data_path)

print(model.prior_mean, model.prior_std)

# clfs = load_mnist_svhn_classifiers('../../classifiers')

# test_set = MnistSvhn(split='test', data_multiplication=20)

# output = CoherenceEvaluator(model,clfs,test_set,data_path).eval()
