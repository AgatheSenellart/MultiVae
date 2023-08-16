from typing import Literal

from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class ReconstructionConfig(EvaluatorConfig):
    """

    Config class for a quantitative evaluation of the reconstruction quality.

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

        metric (Literal['SSIM', 'MSE']) : The metric to use to assess reconstruction quality.
            Default to 'SSIM'.


    """

    metric: Literal["SSIM", "MSE"] = "SSIM"
