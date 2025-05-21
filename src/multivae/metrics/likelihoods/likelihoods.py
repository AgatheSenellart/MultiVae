from pythae.models.base.base_utils import ModelOutput

from ..base.evaluator_class import Evaluator
from .likelihoods_config import LikelihoodsEvaluatorConfig

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from multivae.data.utils import set_inputs_to_device


class LikelihoodsEvaluator(Evaluator):
    """Class for computing likelihood metrics.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt
            file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the
            evaluation.
    """

    def __init__(
        self, model, test_dataset, output=None, eval_config=LikelihoodsEvaluatorConfig()
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        self.num_samples = eval_config.num_samples
        self.batch_size_k = eval_config.batch_size_k
        self.unified = eval_config.unified_implementation

    def eval(self):
        self.joint_nll()
        self.log_to_wandb()
        return ModelOutput(**self.metrics)

    def joint_nll(self):
        ll = 0
        for batch in tqdm(self.test_loader):
            batch = set_inputs_to_device(batch, self.device)
            if self.unified or (not hasattr(self.model, "compute_joint_nll_paper")):
                ll += self.model.compute_joint_nll(
                    batch, self.num_samples, self.batch_size_k
                )
            else:
                self.logger.info("Using the paper version of the joint nll.")
                ll += self.model.compute_joint_nll_paper(
                    batch, self.num_samples, self.batch_size_k
                )

        joint_nll = ll / len(self.test_loader.dataset)
        self.logger.info(f"Mean Joint likelihood : {str(joint_nll)}")
        self.metrics["joint_likelihood"] = joint_nll
        return joint_nll

    def joint_nll_from_subset(self, subset):
        """Only available for the MoPoE model for now. Use a subset posterior
        instead of the joint posterior as the importance sampling distribution.
        """
        if hasattr(self.model, "_compute_joint_nll_from_subset_encoding"):
            ll = 0
            nb_batch = 0
            for batch in self.test_loader:
                batch = set_inputs_to_device(batch, self.device)
                ll += self.model._compute_joint_nll_from_subset_encoding(
                    subset, batch, self.num_samples, self.batch_size_k
                )
                nb_batch += 1

            joint_nll = ll / self.n_data
            self.logger.info("Joint likelihood from subset %s", str(joint_nll))
            self.metrics[f"Joint likelihood from subset {subset}"] = joint_nll
            return joint_nll
        else:
            return None
