from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MVAEConfig(BaseMultiVAEConfig):

    """
    Config class for the MVAE model from 'Multimodal Generative Models for Scalable Weakly-Supervised Learning'.
    https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html

    Args :
        k (int) : The number of subsets to use in the objective. The MVAE objective is the sum
            of the unimodal ELBOs, the joint ELBO and of k random subset ELBOs. Default to 1.
        warmup (int) : If warmup > 0, the MVAE model uses annealing during the first warmup epochs.
            In the objective, the KL terms are weighted by a factor beta that is linearly brought
            to 1 during the first warmup epochs. Default to 10.


    """

    k: int = 0
    warmup: int = 10
