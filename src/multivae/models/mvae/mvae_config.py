from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MVAEConfig(BaseMultiVAEConfig):

    """
    Config class for the MVAE model from 'Multimodal Generative Models for Scalable Weakly-Supervised Learning'.
    https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        recon_losses (Dict[str, Union[function, str]]). The reconstruction loss to use per modality.
            Per modality, you can provide a string in ['mse','bce','l1']. If None is provided, an Mean-Square-Error (mse)
            is used for each modality.
        k (int) : The number of subsets to use in the objective. The MVAE objective is the sum
            of the unimodal ELBOs, the joint ELBO and of k random subset ELBOs. Default to 1.
        warmup (int) : If warmup > 0, the MVAE model uses annealing during the first warmup epochs.
            In the objective, the KL terms are weighted by a factor beta that is linearly brought
            to 1 during the first warmup epochs. Default to 10.


    """

    k: int = 0
    warmup: int = 10
