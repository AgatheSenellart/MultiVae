from itertools import combinations

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE

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

    def __init__(
        self, model, test_dataset, output=None, eval_config=LikelihoodsEvaluatorConfig()
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        self.num_samples = eval_config.num_samples
        self.batch_size_k = eval_config.batch_size_k

    def eval(self):
        joint = self.joint_nll()
        joint_from_sub = self.joint_nll_from_subset(list(self.model.encoders.keys()))
        return ModelOutput(
            joint_likelihood=joint, joint_likelihood_from_subset_expr=joint_from_sub
        )

    def joint_nll(self):
        ll = 0
        nb_batch = 0
        for batch in self.test_loader:
            batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
            ll += self.model.compute_joint_nll(batch, self.num_samples, self.batch_size_k)
            nb_batch += 1

        joint_nll = ll / nb_batch
        self.logger.info(f"Joint likelihood : {str(joint_nll)}")

        return joint_nll

    def joint_nll_from_subset(self, subset):
        if hasattr(self.model, "compute_joint_nll_from_subset_encoding"):
            ll = 0
            nb_batch = 0
            for batch in self.test_loader:
                batch.data = {m: batch.data[m].to(self.device) for m in batch.data}
                ll += self.model.compute_joint_nll_from_subset_encoding(
                    subset, batch, self.num_samples, self.batch_size_k
                )
                nb_batch += 1

            joint_nll = ll / nb_batch
            self.logger.info(
                f"Joint likelihood from subset {subset} : {str(joint_nll)}"
            )
            return joint_nll
        else:
            return None
