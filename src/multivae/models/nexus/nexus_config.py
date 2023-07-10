from pydantic.dataclasses import dataclass
from ..base import BaseMultiVAEConfig
from typing import Dict, Literal

@dataclass
class NexusConfig(BaseMultiVAEConfig):
    """
    This is the base config for the Nexus model from
    "Leveraging hierarchy in multimodal generative models for effective cross-modality inference" (Vasco et al 2022)

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the higher level latent space called z_{\sigma} in the paper. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. It is used in a number of models
            which is why we include it here. Default to False.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. If None is provided,
            a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}
        modalities_specific_dim (Dict[int]) : dimensions of the first level latent variables for all modalities,
            noted at z^(i) in the paper. Default to None
        alpha (float) : hyperparameter that scales the top level KL divergences.
        beta (dict[str, float]) : hyperparameters that scales the bottom modality-specific KL divergence.
        top_level_scalings (dict[str, float]) : hyperparameters that scales the modality specific representation
            reconstruction terms.
        dropout_rate (float between 0 and 1) : dropout rate of the modalities during training. Default to 0.
        msg_dim (int) : Dimension of the messages from each modality. Default to 10.
        
        
    """

    modalities_specific_dim : Dict[str, int] = None
    alpha : float = 1
    beta : Dict[str, float] = None
    top_level_scalings : Dict[str, float] = None
    dropout_rate : float = 0
    msg_dim : int = 10
    