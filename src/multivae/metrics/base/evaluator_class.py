import logging

import torch
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE

from .evaluator_config import EvaluatorConfig


class Evaluator:
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
        model: BaseMultiVAE,
        test_dataset: MultimodalBaseDataset,
        output: str = None,
        eval_config=EvaluatorConfig(),
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.n_data = len(test_dataset)
        self.batch_size = eval_config.batch_size
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=eval_config.batch_size)
        self.set_logger(output)

    def set_logger(self, output):
        logger = logging.getLogger()
        logger.setLevel(logging.NOTSET)

        # our first handler is a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # the second handler is a file handler
        if output is not None:
            file_handler = logging.FileHandler(output + "/metrics.log")
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        self.logger = logger
