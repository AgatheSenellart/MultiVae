from typing import Dict, List, Literal, Union

from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class NexusConfig(BaseMultiVAEConfig):
    """This is the base config for the Nexus model from (Vasco et al 2022).

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        rescale_factors (dict[str, float]): The reconstruction rescaling factors per modality.
            If None is provided but uses_likelihood_rescaling is True, a default value proportional to the input modality
            size is computed. Default to None.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. For Bernoulli distribution,
            the decoder is expected to output **logits**. If None is provided, a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary with
            :code:`decoder_dist_params =  {'mod1' : {"scale" : 0.75}}`.
        modalities_specific_dim (Dict[int]) : dimensions of the first level latent variables for all modalities,
            noted at z^(i) in the paper. Default to None
        bottom_betas (dict[str, float]) : hyperparameters that scales the bottom modality-specific KL divergence.
        dropout_rate (float between 0 and 1) : dropout rate of the modalities during training. Default to 0.
        msg_dim (int) : Dimension of the messages from each modality. Default to 10.
        aggregator (Literal['mean']): Default to 'mean'
        top_beta (float): parameter that scales the KL of the higher level ELBO. Default to 1.
        gammas (Dict[str, float]). Default to None. Factors that rescale the reconstruction of
            each top-level representation of each modality.
        rescale_factors (Dict[str, float]). Default to None. Factors that rescale the reconstruction of
            each modality. Correspond to the lambda factors in the appendix of the paper.
        warmup (int) : number of epochs for the annealing of the KL terms in the loss. Default to 20.
        adapt_top_decoder_variance (List['str']) : For the listed modalities adapt the scale of the top decoders using the procedure
            cited in https://arxiv.org/pdf/2006.13202 . Default to [].
    """

    modalities_specific_dim: Dict[str, int] = None
    bottom_betas: Union[Dict[str, float], None] = None
    dropout_rate: float = 0
    msg_dim: int = 10
    aggregator: Literal["mean"] = "mean"
    top_beta: float = 1
    gammas: Union[Dict[str, float], None] = None
    warmup: int = 20
    adapt_top_decoder_variance: Union[List[str], None] = None
