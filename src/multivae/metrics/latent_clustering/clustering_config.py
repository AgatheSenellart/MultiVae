from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig
from typing import Literal

@dataclass
class ClusteringConfig(EvaluatorConfig):

    """

    Config class for the visualization module.

     Args :
         batch_size (int) : The batch size to use in the evaluation.
         wandb_path (str) : The user can provide the path of the wandb run with a
             format 'entity/projet_name/run_id' where the metrics should be logged.
             For an existing run (the training run), the info can be found in the training dir (in wandb_info.json)
             at the end of training (if wandb was used) or on the hugging_face webpage of the run.
             Otherwise the user can create a new wandb run and get the path with :
                 `
                 import wandb
                 run = wandb.init(entity = your_entity, project=your_project)
                 wandb_path = run.path

             If None are provided, the metrics are not logged on wandb.
             Default to None.
         clustering_method (Literal['kmeans']) :  The method to use to cluster.
         n_clusters (int) :the number of clusters. Default to 10.
         number_of_runs (int) : When computing accuracies, how many runs of clustering to perform to 
            to compute the average accuracies. Default to 20.
         num-samples_for_fit (int) : Number of training samples to use to fit the clusters.
         use_mean (bool) : whether to use a sample or the mean of the encoding distribution. Default to True.

    """

    clustering_method : Literal['kmeans'] = 'kmeans'
    n_clusters : int = 10
    number_of_runs: int = 20
    num_samples_for_fit : int = None
    use_mean : bool = True
