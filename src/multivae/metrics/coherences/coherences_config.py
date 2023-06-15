from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class CoherenceEvaluatorConfig(EvaluatorConfig):
    """

    Config class for the evaluation of the coherences module.

    Args:
        batch_size (int) : The batch size to use in the evaluation.
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            For an existing run (the training run), the info can be found in the training dir (in wandb_info.json)
            at the end of training (if wandb was used) or on the hugging_face webpage of the run.
            Otherwise the user can create a new wandb run and get the path with :

                .. code-block:: python

                    >>> import wandb
                    >>> run = wandb.init(entity = your_entity, project=your_project)
                    >>> wandb_path = run.path


            If None are provided, the metrics are not logged on wandb.
            Default to None.
        include_recon (bool) : If True, we include the reconstructions in the mean conditional generations
            coherences. Default to False.
        nb_samples_for_joint (int): How many samples to use to compute joint coherence. Default to 10000.


    """

    include_recon: bool = False
    nb_samples_for_joint: int = 10000
