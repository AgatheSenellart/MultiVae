from pythae.models.base.base_utils import ModelOutput

from multivae.data import MultimodalBaseDataset

from ..base.evaluator_class import Evaluator
from .likelihoods_config import LikelihoodsEvaluatorConfig

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x


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
        self.unified = eval_config.unified_implementation

    def eval(self):
        self.joint_nll()
        self.log_to_wandb()
        return ModelOutput(**self.metrics)

    def joint_nll(self):
        ll = 0
        for batch in tqdm(self.test_loader):
            batch = MultimodalBaseDataset(
                data={m: batch["data"][m].to(self.device) for m in batch["data"]}
            )
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
        self.logger.info(f"Joint likelihood : {str(joint_nll)}")
        self.metrics["joint_likelihood"] = joint_nll
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

            joint_nll = ll / self.n_data
            self.logger.info(
                f"Joint likelihood from subset {subset} : {str(joint_nll)}"
            )
            self.metrics[f"Joint likelihood from subset {subset}"] = joint_nll
            return joint_nll
        else:
            return None

    def cond_nll_from_subset(self, subset, pred_mods):
        pass

    def reproduce_mopoe_graph(self):
        """
        Computes all the likelihoods from a subset of modalities.
        """

        modalities = list(self.model.encoders.keys())
        liks = []
        for n in range(1, self.model.n_modalities + 1):
            subsets_of_size_n = combinations(
                modalities,
                n,
            )
            liks.append([])
            for s in subsets_of_size_n:
                s = list(s)
                mean_joint = self.joint_nll_from_subset(s)
                liks[-1].append(mean_joint)
        mean_liks = [np.mean(l) for l in liks]
        std_liks = [np.std(l) for l in liks]

        for i in range(len(mean_liks)):
            self.logger.info(
                f"Conditional accuracies for {i+1} modalities : {mean_liks[i]} +- {std_liks[i]}"
            )
            self.metrics[f"Conditional accuracies for {i+1} modalities"] = mean_liks[i]
            self.metrics[
                f"Conditional accuracies for {i+1} modalities (std)"
            ] = std_liks[i]

        self.log_to_wandb()

        return mean_liks, std_liks
