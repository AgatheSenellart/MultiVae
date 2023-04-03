'''
Plot the accuracies as a function of the number of input modalities for different models.
'''

from multivae.models import AutoModel
from multivae.metrics import CoherenceEvaluator
import numpy as np
from classifiers import load_mmnist_classifiers
from multivae.data.datasets.mmnist import MMNISTDataset
import matplotlib.pyplot as plt

clfs = load_mmnist_classifiers()
test_set = MMNISTDataset(data_path="../../../data/MMNIST", split="test")

models = {
    # 'JNF' : ['asenella/mmnistJNF_config1_'],
    # 'JNFDcca' : ['asenella/mmnistJNFDcca_config1_'],
    'MVTCAE' : ['asenella/mmnistMVTCAE_config1_']
}

results = {}
for model_name in models:
    results[model_name] = np.zeros((len(models[model_name]),5))
    for i,model_instance in enumerate(models[model_name]):
        model = AutoModel.load_from_hf_hub(model_instance, allow_pickle=True)
        acc = CoherenceEvaluator(model, clfs,test_set,output=None).eval()
        results[model_name][i][0] += acc.joint_coherence
        results[model_name][i][1:] += np.array(acc.means_coherences)

# Plots
fig = plt.figure()

for m in results:
    values = np.mean(results[m],axis=0)
    stds = np.std(results[m], axis=0)
    plt.errorbar(x=['Joint','1','2','3','4'],y=values,yerr=stds,label=m)

plt.title('Coherences of generations')
plt.legend()
plt.savefig('compare_mmnist.png')
    

