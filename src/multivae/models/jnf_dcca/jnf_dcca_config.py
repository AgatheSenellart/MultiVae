from pydantic.dataclasses import dataclass

from ..joint_models import BaseJointModelConfig


@dataclass
class JNFDccaConfig(BaseJointModelConfig):
    """
    This is the base config for the JNFDcca model.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        uses_likelihood_rescaling (bool): To mitigate modality collapse, it is possible to use likelihood rescaling.
            (see : https://proceedings.mlr.press/v162/javaloy22a.html).
            The inputs_dim must be provided to compute the likelihoods rescalings. Default to False.
        decoders_dist (Dict[str, Union[function, str]]). The decoder distributions to use per modality.
            Per modality, you can provide a string in ['normal','bernoulli','laplace']. If None is provided,
            a normal distribution is used for each modality.
        decoder_dist_params (Dict[str,dict]) : Parameters for the output decoder distributions, for
            computing the log-probability.
            For instance, with normal or laplace distribution, you can pass the scale in this dictionary.
            ex :  {'mod1' : {scale : 0.75}}
        warmup (int): The number of warmup epochs during training. Default to 10.

        nb_epochs_dcca (int) : The number of epochs during which to train the DCCA embeddings. Default to 30.
        embedding_dcca_dim (int) : The dimension of the DCCA embedding to use. Default to 20.
        use_all_singular_values (bool) : Wether to use all the singular values for the computation of the objective.
            Using True is more unstable. Default to False.

    """

    warmup: int = 10
    nb_epochs_dcca: int = 30
    embedding_dcca_dim: int = 20
    use_all_singular_values: bool = (
        False  # Using True generally leads to NaN in the loss.
    )
