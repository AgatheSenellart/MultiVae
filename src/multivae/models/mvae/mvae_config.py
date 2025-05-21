from pydantic.dataclasses import dataclass

from ..base import BaseMultiVAEConfig


@dataclass
class MVAEConfig(BaseMultiVAEConfig):
    """Config class for the MVAE model from 'Multimodal Generative Models for Scalable Weakly-Supervised Learning'.
    https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html.

    Args:
        n_modalities (int): The number of modalities. Default: None.
        latent_dim (int): The dimension of the latent space. Default: None.
        input_dims (dict[str,tuple]): The modalities'names (str) and input shapes (tuple).
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
        use_subsampling (bool) : If True, we use the subsampling paradigm described in the article, not only taking the
            joint ELBO but also the unimodal ELBOs and k random subset elbos. This is useful when training with
            a complete dataset but should be set to False when the dataset is already incomplete.
            Default to True.
        k (int): The number of subsets to use in the objective. The MVAE objective is the sum
            of the unimodal ELBOs, the joint ELBO and of k random subset ELBOs. Default to 0.
        warmup (int): If warmup > 0, the MVAE model uses annealing during the first warmup epochs.
            In the objective, the KL terms are weighted by a factor beta that is linearly brought
            to 1 during the first warmup epochs. Default to 10.
        beta (float): The scaling factor for the divergence term. Default to 1.


    """

    use_subsampling: bool = True
    k: int = 0
    warmup: int = 10
    beta: float = 1
