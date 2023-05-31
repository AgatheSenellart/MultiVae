"""
Plot the accuracies as a function of the number of input modalities for different models.
"""

import matplotlib.pyplot as plt
import numpy as np
from classifiers import load_mmnist_classifiers

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.metrics import FIDEvaluator
from multivae.models import AutoModel

test_set = MMNISTDataset(data_path="../../../data", split="test")

models = {
    "JNF": ["asenella/mmnistJNF_config2_"],
    "JNFDcca": ["asenella/mmnistJNFDcca_config2_"],
    "MVTCAE": ["asenella/mmnistMVTCAE_config2_"],
    "MOPOE": ["asenella/mmnistMoPoE_config2_"],
}

results = {}
for model_name in models:
    print(f"Computing FIDs for model {model_name}")
    results[model_name] = np.zeros((len(models[model_name]), 4))
    for i, model_instance in enumerate(models[model_name]):
        model = AutoModel.load_from_hf_hub(model_instance, allow_pickle=True)
        fds = FIDEvaluator(model, test_set, output="./").mvtcae_reproduce_fids("m0")
        results[model_name][i] += fds.fids

# config1
# results = {
#     'JNF' : [[135.28582310395564,130.77103066549603,140.81823271665206,149.39137339771003]],
#     'JNF-DCCA' : [[117.773292188227,118.87162060165463,130.08200163956542,139.1528229310688]],
#     'MVTCAE' : [[106.54300877505386,106.55273888671888,116.93112002118824,128.96500577173066]]
# }


# Plots
fig = plt.figure()

for m in results:
    values = np.mean(results[m], axis=0)
    stds = np.std(results[m], axis=0)
    plt.errorbar(x=["1", "1,2", "1,2,3", "1,2,3,4"], y=values, yerr=stds, label=m)

plt.title("FIDs of generations")
plt.legend()
plt.savefig("compare_mmnist.png")
