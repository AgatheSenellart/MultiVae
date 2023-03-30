import os
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
        self.batch_size = eval_config.batch_size
        self.test_loader = DataLoader(test_dataset, batch_size=eval_config.batch_size)
        if output is not None:
            if not os.path.exists(output + "/metrics.txt"):
                open(output + "/metrics.txt", "w+")
            self.f = open(output + "/metrics.txt", "a")
            print("Writing results in ", self.f)



    

        


