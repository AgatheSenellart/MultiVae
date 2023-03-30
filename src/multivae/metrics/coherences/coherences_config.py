from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class CoherenceEvaluatorConfig(EvaluatorConfig):
    """

    Config class for the evaluation of the coherences module.

    Args :
        batch_size (int) : The batch size to use in the evaluation.
        include_recon (bool) : If True, we include the reconstructions in the mean conditional generations
            coherences.


    """

    include_recon: bool = False
    nb_samples_for_joint: int = 10000
