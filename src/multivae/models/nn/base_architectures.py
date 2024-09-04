from pythae.models.base.base_model import BaseEncoder




class BaseJointEncoder(BaseEncoder):
    """This is a base class for Encoders neural networks."""

    def __init__(self):
        BaseEncoder.__init__(self)
        self.latent_dim  = None # to be set in child classs

    def forward(self, x : dict):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own joint encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Joint_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...         self.latent_dim = ...
            ...
            ...     def forward(self, x: dict):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (dict): Multimodal input to encode : a dictionary that contains modalities' names as keys and modalities' data as values.

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()
