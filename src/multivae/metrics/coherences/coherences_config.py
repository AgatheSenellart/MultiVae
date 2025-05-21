from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class CoherenceEvaluatorConfig(EvaluatorConfig):
    """Config class for the evaluation of the coherences module.

    Args:
        batch_size (int) : The batch size to use in the evaluation.
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            See :doc:`info_wandb` for more information.
            If None is provided, the metrics are not logged on wandb.
            Default to None.
        num_classes (int) : Number of Classes. Default to 10.
        include_recon (bool) : If True, we include the reconstructions in the mean conditional generations
            coherences. Default to False.
        nb_samples_for_joint (int): How many samples to use to compute joint coherence. Default to 10000.
        nb_samples_for_cross (int): How many generations *per sample* to use when computing cross coherences. Default to 1.
        give_details_per_class (bool) : Provide accuracy details per class. Default to False.


    """

    num_classes: int = 10
    include_recon: bool = False
    nb_samples_for_joint: int = 10000
    nb_samples_for_cross: int = 1
    give_details_per_class: bool = False
