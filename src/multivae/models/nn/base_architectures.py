from typing import Dict

import torch
from pythae.models.base.base_model import BaseDecoder, BaseEncoder


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
        below.

        .. code-block::

            >>> from multivae.models.nn import BaseEncoder
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
            ...         # x is a dict with a tensor for each modality
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Args:
            x (dict): Multimodal input to encode : a dictionary that contains modalities' names as keys and modalities' data as values.

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseMultilatentEncoder(BaseEncoder):
    """This is a base class for for encoders with multiple latent spaces."""

    def __init__(self):
        BaseEncoder.__init__(self)
        self.latent_dim = None  # to be set in child class
        self.style_dim = None  # to be set in child class

    def forward(self, x: torch.Tensor):
        r"""This function must be implemented in a child class.
        It takes the input tensor x and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput` with the parameters for the shared latent space
        and the modality-specific latent space.
        If you decide to provide your own encoder network in a model that uses multiple
        latent spaces, you must make sure your
        model inherits from this class by setting and then defining your forward function as
        below.

        .. code-block::

            >>> from multivae.models.nn import BaseMultilatentEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseMultilatentEncoder):
            ...
            ...     def __init__(self):
            ...         BaseMultilatentEncoder.__init__(self)
            ...         # your code
            ...         self.latent_dim = ...
            ...         self.style_dim = ...
            ...
            ...     def forward(self, x):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding= ..., # shared latent space
                            log_covariance=...,
                            style_embedding=..., # modality-specific latent space
                            style_log_covariance=...,
            ...         )
            ...         return output

        Args:
            x (torch.Tensor): Input data

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder.
        """
        raise NotImplementedError()


class BaseConditionalDecoder(BaseDecoder):
    """This is a base class for Conditional Decoders architectures."""

    def __init__(self):
        BaseDecoder.__init__(self)
        self.latent_dim = None  # to be set in child class

    def forward(self, z: torch.Tensor, cond_mods: Dict[str, torch.Tensor]):
        r"""This function must be implemented in a child class.
        It takes the latent variable z and conditioning modality and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput` with the reconstruction.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        below.

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
            ...     def forward(self, z, cond_mods):
            ...         # your code
            ...         output = ModelOutput(
            ...             reconstruction= ...
            ...         )
            ...         return output

        Args:
            z (torch.Tensor): Latent variable
            cond_mods (Dic[str, torch.Tensor]): Conditioning data.

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder.
        """
        raise NotImplementedError()
