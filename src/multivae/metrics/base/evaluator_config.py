from pydantic.dataclasses import dataclass
from pythae.models.base.base_config import BaseConfig


@dataclass
class EvaluatorConfig(BaseConfig):
    """

    Base config class for the evaluation modules.

    Args :
        batch_size (int) : The batch size to use in the evaluation.
        wandb_path (str) : The path of the wandb run with a format 'entity/projet_name/run_id'.
            If one is provided, the metrics are also logged on wandb. 

    """

    batch_size: int = 512
    wandb_path: str = None
    
