from ..base import BaseMultiVAEConfig


class MMVAEConfig(BaseMultiVAEConfig):

    """

    Args :
        K (int) : the number of samples to use in the DreG loss. Default to 1.
        use_likelihood_rescaling (bool) : Use likelihood rescaling to mitigate modality collapse.
            Default to True.
        posterior_dist : (str) Default to 'laplace with softmax' the posterior distribution that is used in
            the original paper. Possible values ['laplace_with_softmax','normal']

    """

    K: int = 10
    posterior_dist = "laplace_with_softmax"
    learn_prior = True
