from typing import Literal

from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class ReconstructionConfig(EvaluatorConfig):
    """Config class for a quantitative evaluation of the reconstruction quality.

    Args:
        batch_size (int) : The batch size to use in the evaluation.
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            See :doc:`info_wandb` for more information.
            If None is provided, the metrics are not logged on wandb.
            Default to None.

        metric (Literal['SSIM', 'MSE']) : The metric to use to assess reconstruction quality.
            Default to 'SSIM'.


    """

    metric: Literal["SSIM", "MSE"] = "SSIM"
