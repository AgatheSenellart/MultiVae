"""
Plot the accuracies as a function of the number of input modalities for different models.
"""

import matplotlib.pyplot as plt
import numpy as np
from classifiers import load_mmnist_classifiers

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import CoherenceEvaluator
from multivae.models import AutoModel

# clfs = load_mmnist_classifiers()
# test_set = MMNISTDataset(data_path="../../../data", split="test")

# models = {
#     # 'JNF' : ['asenella/mmnistJNF_config1_'],
#     # 'JNFDcca' : ['asenella/mmnistJNFDcca_config1_'],
#     'MVTCAE' : ['asenella/mmnistMVTCAE_config1_']
# }

# results = {}
# for model_name in models:
#     results[model_name] = np.zeros((len(models[model_name]),5))
#     for i,model_instance in enumerate(models[model_name]):
#         model = AutoModel.load_from_hf_hub(model_instance, allow_pickle=True)
#         acc = CoherenceEvaluator(model, clfs,test_set,output=None).eval()
#         results[model_name][i][0] += acc.joint_coherence
#         results[model_name][i][1:] += np.array(acc.means_coherences)

# config1
results = {
    "JNF": [[0.08, 0.65, 0.76, 0.84, 0.85]],
    "JNFDcca": [[0.09, 0.76, 0.85, 0.88, 0.88]],
    "MVTCAE": [[0.005, 0.46, 0.63, 0.71, 0.75]],
}

# config2
# results = {
#     'JNF' : [[0.03,0.66,0.79,0.86,0.88]],
#     'MVTCAE' : [[0.003,0.61, 0.78,0.85,0.87]],
#     'JNFDccA' : [[0.04,0.73,0.85,0.90,0.92]],
#     'MoPoE' : [[0.08,0.60,0.72,0.77,0.80]]
# }

# reproduce_mopoe
# results = {
#     'MoPoE' : [[0.11,0.62,0.75,0.79,0.80]],
#     'MVTCAE' : [[0.003,0.56,0.75,0.82,0.85]]
# }

# Plots
fig = plt.figure()

for m in results:
    values = np.mean(results[m], axis=0)
    stds = np.std(results[m], axis=0)
    plt.errorbar(x=["Joint", "1", "2", "3", "4"], y=values, yerr=stds, label=m)

plt.title("Coherences of generations")
plt.legend()
plt.savefig("compare_mmnist.png")
