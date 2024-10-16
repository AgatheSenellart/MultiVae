from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig
from typing import Dict, Literal

@dataclass
class JNFGMCConfig(BaseJointModelConfig):
    """
    This is the base config for the JNFGMC model.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. Default to False.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. For Bernoulli distribution,
            the decoder is expected to output **logits**. If None is provided, a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}
        warmup (int): The number of warmup epochs during training. Default to 10.

        annealing (bool) : Whether to train the unimodal encoders and joint in one step using annealing. 
            In that case, the parameter alpha is used. 
            
        alpha (float) : the weight that ponders the unimodal encoders terms in the joint loss. Default to 0.1.
        
        nb_epochs_gmc (int) : The number of epochs during which to train the DCCA embeddings. Default to 30.
        
        beta (float) :Weighing factor for the regularization of the joint VAE. Default to 1.
        
        logits_to_std (Literal['standard', 'softplus']). Default to 'standard'. How to compute the std deviations from the logits outputs from the 

    """

    warmup: int = 10
    annealing: bool = False
    alpha: float = 0.1
    nb_epochs_gmc: int = 30
    beta: float = 1.0
    gmc_config : Dict = None
    logits_to_std : Literal['standard', 'softplus'] = 'standard'
