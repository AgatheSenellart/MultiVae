from typing import Literal

from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class ClusteringConfig(EvaluatorConfig):
    """Config class for the clustering module.

    Args:
        batch_size (int) : The batch size to use in the evaluation. Default to 512
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            See :doc:`info_wandb` for more information.
            If None is provided, the metrics are not logged on wandb.
            Default to None.
        clustering_method (Literal['kmeans']) :  The method to use to cluster. Default to 'kmeans'
        n_clusters (int) :the number of clusters. Default to 10.
        number_of_runs (int) : When computing accuracies, how many runs of clustering to perform to
        to compute the average accuracies. Default to 20.
        num-samples_for_fit (int) : Number of training samples to use to fit the clusters. If None,
            uses all the samples. Default to None.
        use_mean (bool) : Whether to use a sample or the mean of the encoding distribution as the
            representative embedding. Default to True.

    """

    clustering_method: Literal["kmeans"] = "kmeans"
    n_clusters: int = 10
    number_of_runs: int = 20
    num_samples_for_fit: int = None
    use_mean: bool = True
