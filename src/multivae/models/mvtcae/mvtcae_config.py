from pydantic.dataclasses import dataclass

from ..base.base_config import BaseMultiVAEConfig


@dataclass
class MVTCAEConfig(BaseMultiVAEConfig):
    """
    This is the base config class for the MVTCAE model from
    'Multi-View Representation Learning via Total Correlation Objective' Neurips 2021.
    The code is based on the original implementation that can be found here :
    https://github.com/gr8joo/MVTCAE/blob/master/run_epochs.py

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
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
        alpha (float) : The parameter that ponderates the total correlation ratio in the loss.
        beta (float) : The parameter that weights the sum of all KLs
    """

    alpha: float = 0.1
    beta: float = 2.5
