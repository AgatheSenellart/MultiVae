from typing import Dict, List, Union

from pydantic.dataclasses import dataclass

from ..base.base_config import BaseMultiVAEConfig


@dataclass
class MoPoEConfig(BaseMultiVAEConfig):
    """
    This class is the configuration class for the MoPoE model, from
    'Generalized Multimodal ELBO' Sutter 2021 (https://arxiv.org/abs/2105.02470)


    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. Default to True.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. If None is provided,
            a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}
        subsets (List[list] or Dict[list]) : List or dictionary containing the subsets to consider. If None is provided,
            all subsets are considered. Examples of valid inputs : [['mod_1', 'mod_2'], ['mod_1'], ['mod_2']]
            or {'s1' : ['mod_1', 'mod_2], 's2' : ['mod_1'], 's3' : ['mod_2']}
            Default to None.
        beta (float) : The weight to the KL divergence term in the ELBO. Default to 1.0
    """

    subsets: Union[Dict[str, list], List[list]] = None
    beta: float = 1.0
    use_modality_specific_spaces : bool = False
    beta_style: float = 1.0
    modalities_specific_dim : dict = None
    
