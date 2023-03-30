from itertools import combinations
import numpy as np
import torch
from torch.utils.data import DataLoader
from multivae.models.base import BaseMultiVAE
from multivae.data import MultimodalBaseDataset
from ..base.evaluator_class import Evaluator
from .likelihoods_config import LikelihoodsEvaluatorConfig

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
    
    def __init__(self, model, test_dataset, output=None, eval_config=LikelihoodsEvaluatorConfig()) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        self.K = eval_config.K
        self.batch_size_k = eval_config.batch_size_k
    
    def eval(self):
        self.joint_nll()
        self.joint_nll_from_subset(list(self.model.encoders.keys()))

    def joint_nll(self):
        ll = 0
        nb_batch = 0
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            ll += self.model.compute_joint_nll(batch,self.K, self.batch_size_k)
            nb_batch += 1

        joint_nll = ll / nb_batch
        if hasattr(self, "f"):
            self.f.write(f"\n Joint likelihood : {str(joint_nll)} \n")
            
            
    def joint_nll_from_subset(self,subset):
        
        if hasattr(self.model, 'compute_joint_nll_from_subset_encoding'):
            ll = 0
            nb_batch = 0
            for batch in self.test_loader:
                batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
                ll += self.model.compute_joint_nll_from_subset_encoding(subset,batch,self.K, self.batch_size_k)
                nb_batch += 1

            joint_nll = ll / nb_batch
            if hasattr(self, "f"):
                self.f.write(f"\n Joint likelihood from subset {subset} : {str(joint_nll)} \n")
