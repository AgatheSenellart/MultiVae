from pythae.models.base.base_model import BaseEncoder, BaseDecoder, ModelOutput
import torch

class BaseJointEncoder(BaseEncoder):
    """This is a base class for Joint Encoders neural networks."""

    def __init__(self):
        BaseEncoder.__init__(self)
        self.latent_dim = None  # to be set in child class

    def forward(self, x: dict):
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
    
    
class BaseConditionalDecoder(BaseDecoder):
    
    """This is a base class for Conditional Decoders architectures.
    """
    
    def __init__(self):
        BaseDecoder.__init__(self)
        self.latent_dim = None # to be set in child class
        
    
    def forward(self, z: torch.Tensor, conditioning_modality: torch.Tensor):
        
        r"""This function must be implemented in a child class.
        It takes the latent variable z and conditioning modality and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput` with the reconstruction.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseConditionalDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Conditional_Decoder(BaseConditionalDecoder):
            ...
            ...     def __init__(self):
            ...         BaseConditionalDecoder.__init__(self)
            ...         # your code
            ...         self.latent_dim = ...
            ...
            ...     def forward(self, z, conditioning_modality):
            ...         # your code
            ...         output = ModelOutput(
            ...             reconstruction= ...
            ...         )
            ...         return output

        Parameters:
            z (torch.Tensor): Latent variable
            conditioning_modality (torch.Tensor): Conditioning modality data

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder
        """
        raise NotImplementedError()