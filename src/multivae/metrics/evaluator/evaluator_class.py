from itertools import combinations
import numpy as np
import torch
from torch.utils.data import DataLoader
from multivae.models.base import BaseMultiVAE
from multivae.data import MultimodalBaseDataset

from .evaluator_config import EvaluatorConfig


class Evaluator():
    """
    Base class for computing metrics. 
    
    Args:
        model (BaseMultiVAE) : The model to evaluate.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        
        
    """
    def __init__(
        self,
        model : BaseMultiVAE,
        test_dataset : MultimodalBaseDataset,
        output : str =None,
        eval_config=EvaluatorConfig(),
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.n_data = len(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=eval_config.batch_size)
        if output is not None:
            self.f = open(output + "/metrics.txt", "w+")
            print("Writing results in ", self.f)

        

class CoherenceEvaluator(Evaluator):
    """
    Class for computing coherences metrics.
    
    Args:
        model (BaseMultiVAE) : The model to evaluate.
        classifiers (dict) : A dictionary containing the pretrained classifiers to use for the coherence evaluation.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        """
    
    def __init__(self, model, classifiers, test_dataset, output=None, eval_config=EvaluatorConfig()) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        self.clfs = classifiers
        for k in self.clfs:
            self.clfs[k] = self.clfs[k].to(self.device)
            
    
    def eval(self):
        
        """ 
        Computes all the coherences from one subset of modalities to another modality. 
        """
        
        modalities = list(self.model.encoders.keys())
        mean_accs = []
        for n in range(1, self.model.n_modalities):
            subsets_of_size_n = combinations(modalities, n) 
            mean_accs.append([])
            for s in subsets_of_size_n:
                _,mean_acc = self.all_accuracies_from_subset(s)
                mean_accs[-1].append(mean_acc)
        mean_accs = [np.mean(l) for l in mean_accs]
        return mean_accs
            
            
        
    def all_accuracies_from_subset(self, subset):
        
        """
        Compute all the coherences generating from the modalities in subset to a modality 
        that is not in subset.

        Returns:
            dict, float : The dictionary of all coherences from subset, and the mean coherence
        """
        
        accuracies = {}
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            batch.labels = batch.labels.to(self.device)
            pred_mods = [m for m in self.model.encoders if m not in subset]
            output = self.model.predict(batch, subset, pred_mods)
            for pred_m in pred_mods:
                preds = self.clfs[pred_m](output[pred_m])
                pred_labels = torch.argmax(preds, dim=1)
                try:
                    accuracies[pred_m] += torch.sum(pred_labels == batch.labels)
                except:
                    accuracies[pred_m] = torch.sum(pred_labels == batch.labels)

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}
        if hasattr(self, "f"):
            self.f.write(f"Subset {subset} accuracies \n")
            self.f.write(acc.__str__() + "\n")
            mean_pair_acc = np.mean(list(acc.values()))
            self.f.write("Mean subset {subset} accuracies : " + str(mean_pair_acc))
        return acc, mean_pair_acc
        

    def pair_accuracies(self):
        accuracies = {}
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            batch.labels = batch.labels.to(self.device)
            for cond_m in self.model.encoders:
                for pred_m in self.model.encoders:
                    output = self.model.predict(batch, cond_m, pred_m)
                    preds = self.clfs[pred_m](output[pred_m])
                    pred_labels = torch.argmax(preds, dim=1)
                    try:
                        accuracies[cond_m + "_" + pred_m] += torch.sum(
                            pred_labels == batch.labels
                        )
                    except:
                        accuracies[cond_m + "_" + pred_m] = torch.sum(
                            pred_labels == batch.labels
                        )

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}
        if hasattr(self, "f"):
            self.f.write("Pair accuracies \n")
            mean_pair_acc = np.mean(list(acc.values()))
            self.f.write(acc.__str__() + "\n")
            self.f.write("Mean pair accuracies" + str(mean_pair_acc))
        return acc

    def all_one_accuracies(self):
        accuracies = {}
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            batch.labels = batch.labels.to(self.device)
            for pred_m in self.model.encoders:
                cond_m = [m for m in self.model.encoders if m != pred_m]
                output = self.model.predict(batch, cond_m, pred_m)
                preds = self.clfs[pred_m](output[pred_m])
                pred_labels = torch.argmax(preds, dim=1)
                try:
                    accuracies[pred_m] += torch.sum(pred_labels == batch.labels)
                except:
                    accuracies[pred_m] = torch.sum(pred_labels == batch.labels)

        acc = {k: accuracies[k].cpu().numpy() / self.n_data for k in accuracies}
        if hasattr(self, "f"):
            self.f.write("All to one accuracies \n")
            mean_pair_acc = np.mean(list(acc.values()))
            self.f.write(acc.__str__() + "\n")
            self.f.write("Mean all-to-one accuracies" + str(mean_pair_acc))
        return acc
    
class LikelihoodsEvaluator(Evaluator):
    """
    Class for computing likelihood metrics.
    
    Args:
        model (BaseMultiVAE) : The model to evaluate.
        classifiers (dict) : A dictionary containing the pretrained classifiers to use for the coherence evaluation.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        """
    
    def __init__(self, model, test_dataset, output=None, eval_config=EvaluatorConfig()) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        
    
    def eval(self):
        self.joint_nll()

    def joint_nll(self):
        ll = 0
        nb_batch = 0
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            ll += self.model.compute_joint_nll(batch)
            nb_batch += 1

        joint_nll = ll / nb_batch
        if hasattr(self, "f"):
            self.f.write(f"\n Joint likelihood : {str(joint_nll)} \n")
