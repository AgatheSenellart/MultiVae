import importlib
import inspect
import logging
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy
from http.cookiejar import LoadError
from typing import Union

import cloudpickle
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from pythae.models.base.base_utils import CPU_Unpickler, ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from torch.nn import functional as F

from ...data.datasets.base import MultimodalBaseDataset
from ..auto_model import AutoConfig
from ..nn.default_architectures import BaseDictDecoders, BaseDictEncoders
from .base_config import BaseConfig, EnvironmentConfig
from .base_utils import hf_hub_is_available, model_card_template

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseModel(nn.Module):
    """Base class for Multimodal models : including Multimodal VAEs and additional models such as Multimodal Fusion / Representation Models.

    Args:
        model_config (BaseConfig): An instance of BaseMultiVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.


    """

    def __init__(
        self,
        model_config: BaseConfig
    ):
        nn.Module.__init__(self)

        self.model_name = "BaseModel"
        self.model_config = model_config
        self.model_custom_architectures = [] 
        

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """
        Main forward pass outputing the model outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``
        """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass


    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``encoders.pkl`` (resp. ``decoders.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """
        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )

        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        for archi in self.model_config.custom_architectures:
            with open(os.path.join(dir_path, archi + ".pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(
                    inspect.getmodule(self.__getattr__(archi))
                )
                cloudpickle.dump(self.__getattr__(archi), fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = AutoConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model.pt" not in file_list:
            raise FileNotFoundError(
                f"Missing model weights file ('model.pt') file in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        try:
            model_weights = torch.load(path_to_model_weights, map_location="cpu")

        except RuntimeError:
            RuntimeError(
                "Enable to load model weights. Ensure they are saves in a '.pt' format."
            )

        if "model_state_dict" not in model_weights.keys():
            raise KeyError(
                "Model state dict is not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}"
            )

        model_weights = model_weights["model_state_dict"]

        return model_weights

    @classmethod
    def _load_custom_archi_from_folder(cls, dir_path, archi: str):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if archi + ".pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing architecture pkl file ('{archi}.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom encoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, f"{archi}.pkl"), "rb") as fp:
                archi = CPU_Unpickler(fp).load()

        return archi

    @classmethod
    def load_from_folder(cls, dir_path: str):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoders.pkl`` (resp.
                ``decoders.pkl``) if a custom encoders (resp. decoders) were provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        custom_architectures = {}
        for archi in model_config.custom_architectures:
            custom_architectures[archi] = cls._load_custom_archi_from_folder(
                dir_path, archi
            )

        model = cls(model_config, **custom_architectures)
        model.load_state_dict(model_weights)

        return model

    @classmethod
    def _check_python_version_from_folder(cls, dir_path: str):
        if "environment.json" in os.listdir(dir_path):
            env_spec = EnvironmentConfig.from_json_file(
                os.path.join(dir_path, "environment.json")
            )
            python_version = env_spec.python_version
            python_version_minor = python_version.split(".")[1]

            if python_version_minor == "8" and sys.version_info[1] > 8:
                raise LoadError(
                    "Trying to reload a model saved with python3.8 with python3.9+. "
                    "Please create a virtual env with python 3.8 to reload this model."
                )

            elif int(python_version_minor) >= 9 and sys.version_info[1] == 8:
                raise LoadError(
                    "Trying to reload a model saved with python3.9+ with python3.8. "
                    "Please create a virtual env with python 3.9+ to reload this model."
                )
