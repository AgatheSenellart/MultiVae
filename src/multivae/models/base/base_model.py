
import os
from copy import deepcopy

import dill
import torch
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from .base_config import BaseMultiVAEConfig


class BaseMultiVAE(nn.Module):
    """Base class for Multimodal VAE models.

    Args:
        model_config (BaseMultiVAEConfig): An instance of BaseMultiVAEConfig in which any model's parameters is
            made available.

        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each 
            modality (instance of Pythae's BaseEncoder). 

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the encoders for each 
            modality (instance of Pythae's BaseEncoder). 

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: BaseMultiVAEConfig,
        encoders: dict,
        decoders: dict,
    ):

        nn.Module.__init__(self)

        self.model_name = "BaseMultiVAE"

        self.n_modalities = model_config.n_modalities
        
        if self.n_modalities == len(encoders.keys):
            raise AttributeError(
                f"The provided number of encoders {len(encoders.keys)} doesn't"
                "match the number of modalities ({self.n_modalities} in model config "
            )
        
        if self.n_modalities == len(decoders.keys):
            raise AttributeError(
                f"The provided number of decoders {len(decoders.keys)} doesn't"
                "match the number of modalities ({self.n_modalities} in model config "
            )
        
        self.latent_dim = model_config.latent_dim
        self.model_config = model_config


        self.set_decoders(decoders)
        self.set_encoders(encoders)

        self.device = None

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Main forward pass outputing the VAE outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``"""
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def save(self, dir_path):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        model_path = dir_path

        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path)

            except FileNotFoundError as e:
                raise e

        self.model_config.save_json(model_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(model_path, "encoder.pkl"), "wb") as fp:
                dill.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(model_path, "decoder.pkl"), "wb") as fp:
                dill.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))



    def set_encoders(self, encoders: dict) -> None:
        """Set the encoders of the model"""
        for modality in encoders:
            encoder = encoders[modality]
            if not issubclass(type(encoder), BaseEncoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, encoder must inherit from BaseEncoder class from "
                        "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                    )
                )
        self.encoders = encoders

    def set_decoders(self, decoders: dict) -> None:
        """Set the decoders of the model"""
        for modality in decoders:
            decoder = decoders[modality]
            if not issubclass(type(decoder), BaseDecoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, decoder must inherit from BaseDecoder class from "
                        "pythae.models.base_architectures.BaseDecoder. Refer to documentation."
                    )
                )
        self.decoders = decoders
