
import os
from copy import deepcopy

import cloudpickle
import inspect
import torch
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from .base_config import BaseMultiVAEConfig

from ..nn.default_architectures import BaseDictEncoders, BaseDictDecoders

class BaseMultiVAE(nn.Module):
    """Base class for Multimodal VAE models.

    Args:
        model_config (BaseMultiVAEConfig): An instance of BaseMultiVAEConfig in which any model's parameters is
            made available.

        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each 
            modality. Each encoder is an instance of Pythae's BaseEncoder. 

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the decoders for each 
            modality. Each decoder is an instance of Pythae's BaseDecoder. 


    """

    def __init__(
        self,
        model_config: BaseMultiVAEConfig,
        encoders: dict = None,
        decoders: dict = None
    ):

        nn.Module.__init__(self)

        self.model_name = "BaseMultiVAE"
        self.model_config = model_config
        self.n_modalities = model_config.n_modalities
        self.input_dims = model_config.input_dims
        self.model_config.ses_default_encoders = False
        self.model_config.ses_default_decoders = False
        
        if encoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide encoders or input dims for the modalities in the model_config."
                )
            else:
                encoders = BaseDictEncoders(self.input_dims, model_config.latent_dim)
                self.model_config.uses_default_encoders = True
        
        if decoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide decoders or input dims for the modalities in the model_config."
                )
            else:
                decoders = BaseDictDecoders(self.input_dims, model_config.latent_dim)
                self.model_config.uses_default_decoders = True
        
        self.sanity_check(encoders, decoders)
        
        self.latent_dim = model_config.latent_dim
        self.model_config = model_config

        
        self.set_decoders(decoders)
        self.set_encoders(encoders)

        self.device = None
        
    def sanity_check(self, encoders, decoders):
        
        if self.n_modalities != len(encoders.keys()):
            raise AttributeError(
                f"The provided number of encoders {len(encoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )
        
        if self.n_modalities != len(decoders.keys()):
            raise AttributeError(
                f"The provided number of decoders {len(decoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )
            
        if encoders.keys() != decoders.keys():
            raise AttributeError(
                "The names of the modalities in the encoders dict doesn't match the names of the modalities"
                " in the decoders dict."
            )

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """
        Main forward pass outputing the VAE outputs
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



    def set_encoders(self, encoders: dict) -> None:
        """Set the encoders of the model"""
        self.encoders = nn.ModuleDict()
        for modality in encoders:
            encoder = encoders[modality]
            if not issubclass(type(encoder), BaseEncoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, encoder must inherit from BaseEncoder class from "
                        "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                    )
                )
            if encoder.latent_dim != self.latent_dim:
                raise AttributeError(
                    f"The latent dim of encoder {modality} doesn't have the same latent dimension as the "
                    f" model itself ({self.latent_dim})"
                )
            self.encoders[modality] = encoder

    def set_decoders(self, decoders: dict) -> None:
        """Set the decoders of the model"""
        self.decoders = nn.ModuleDict()
        for modality in decoders:
            decoder = decoders[modality]
            if not issubclass(type(decoder), BaseDecoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, decoder must inherit from BaseDecoder class from "
                        "pythae.models.base_architectures.BaseDecoder. Refer to documentation."
                    )
                )
            self.decoders[modality] = decoder

    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``encoders.pkl`` (resp. ``decoders.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoders:
            with open(os.path.join(dir_path, "encoders.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoders))
                cloudpickle.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoders:
            with open(os.path.join(dir_path, "decoders.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoders))
                cloudpickle.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))