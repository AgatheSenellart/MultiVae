import logging

import torch
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE

from .evaluator_config import EvaluatorConfig
from multivae.samplers.base import BaseSampler
from pathlib import Path

class Evaluator:
    """
    Base class for computing metrics.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        sampler (BaseSampler) : A custom sampler for sampling from the common latent space. If None is provided, samples 
            are generated from the prior.


    """

    def __init__(
        self,
        model: BaseMultiVAE,
        test_dataset: MultimodalBaseDataset,
        output: str = None,
        eval_config=EvaluatorConfig(),
        sampler : BaseSampler = None,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.n_data = len(test_dataset)
        self.batch_size = eval_config.batch_size
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=eval_config.batch_size)
        if output is not None:
            Path(output).mkdir(parents=True,exist_ok=True)
        
        self.output = output
            
        self.set_logger(output)
        self.set_wandb(eval_config.wandb_path)
        self.metrics = {}
        self.sampler = sampler
        if self.sampler is not None:
            if not sampler.is_fitted:
               raise AttributeError("The provided sampler is not fitted."
                                     "Please fit the sampler before using it in the evaluator module.")

    def set_logger(self, output):
        logger = logging.getLogger()
        logger.setLevel(logging.NOTSET)

        # our first handler is a console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        logger.addHandler(self.console_handler)

        # the second handler is a file handler
        if output is not None:
            self.file_handler = logging.FileHandler(output + "/metrics.log")
            self.file_handler.setLevel(logging.INFO)
            logger.addHandler(self.file_handler)

        self.logger = logger

    def set_wandb(self, wandb_path):
        
        if wandb_path is None:
            self.wandb_run = None
            return
        else:
            entity, project, run_id = tuple(wandb_path.split("/"))
            try:
                import wandb
            except:
                raise ModuleNotFoundError(
                    "You provided a wandb_path, but `wandb` package is not installed. Run `pip install wandb`"
                )

            self.wandb_run = wandb.init(
                entity=entity, project=project, id=run_id, resume="must", reinit=True
            )
            return

    def log_to_wandb(self):
        if self.wandb_run is not None:
            self.wandb_run.log(self.metrics)
            
    def finish(self):
        """Removes handlers and finish the wandb run. """
        
        self.logger.removeHandler(self.console_handler)
        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)
        if self.wandb_run is not None:
            self.wandb_run.finish()
