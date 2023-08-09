from itertools import combinations
from typing import List

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.samplers.base import BaseSampler

from ..base.evaluator_class import Evaluator
from .reconstruction_config import ReconstructionConfig
from torchmetrics.classification import MulticlassAccuracy

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import TotalVariation as TV



class Reconstruction(Evaluator):
    """
    Class for computing coherences metrics.

    Args:

        model (BaseMultiVAE) : The model to evaluate.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (CoherencesEvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        
    """

    def __init__(
        self,
        model,
        test_dataset,
        output=None,
        eval_config=ReconstructionConfig()
    ) -> None:
        
        super().__init__(model, test_dataset, output, eval_config)
        
        self.metrics_dict = dict(SSIM = SSIM)
        self.metric_name = eval_config.metric
        
        
        

    
    def reconstruction_from_subset(self, subset: List[str]):
        """
        
        Take a subset of modalities as input and compute reconstructions for those 
        modalities.
         
        """

        if self.metric_name in self.metrics_dict :
            metric = self.metrics_dict[self.metric_name]()
            for batch in self.test_loader:
                batch = set_inputs_to_device(batch, self.device)
                output = self.model.predict(batch, list(subset), list(subset))
                for mod in subset:
                    preds = output[mod]
                    target = batch.data[mod]
                    reconstruction_error = metric(preds, target)
            
            mean_recon_error = metric.compute()   
        
        elif self.metric_name=='MSE':  
            mean_recon_error = 0
            n_data = 0
            for batch in self.test_loader:
                batch = set_inputs_to_device(batch, self.device)
                output = self.model.predict(batch, list(subset), list(subset))
                for mod in subset:
                    preds = output[mod]
                    target = batch.data[mod]
                    mean_recon_error += torch.sum((preds-target)**2)
                    n_data += len(preds)
            mean_recon_error = mean_recon_error/n_data
        else :
            raise(AttributeError("Unrecognized metric name for reconstruction error. "))
            
        self.logger.info(f"Subset {subset} reconstruction : {mean_recon_error} ")
        self.metrics.update(
            {f'{subset} reconstruction error ({self.metric_name})' : {mean_recon_error}}
        )
        
        return mean_recon_error

    

    def eval(self):
        
        # Joint reconstruction with all modalities
        self.reconstruction_from_subset(list(self.model.encoders.keys()))
        
        # Unimodal reconstruction
        for mod in self.model.encoders.keys():
            self.reconstruction_from_subset([mod])

        self.log_to_wandb()

        return ModelOutput(**self.metrics)