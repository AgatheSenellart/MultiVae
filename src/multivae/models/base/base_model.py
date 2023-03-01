import inspect
import os
import sys
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
from torch.nn.modules.loss import BCEWithLogitsLoss, L1Loss, MSELoss

from ...data.datasets.base import MultimodalBaseDataset
from ..auto_model import AutoConfig
from ..nn.default_architectures import BaseDictDecoders, BaseDictEncoders
from .base_config import BaseMultiVAEConfig, EnvironmentConfig


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
        decoders: dict = None,
    ):
        nn.Module.__init__(self)

        self.model_name = "BaseMultiVAE"
        self.model_config = model_config
        self.n_modalities = model_config.n_modalities
        self.input_dims = model_config.input_dims
        self.model_config.uses_default_encoders = False
        self.model_config.uses_default_decoders = False

        if encoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide encoders or input dims for the modalities in the model_config."
                )
            else:
                if len(self.input_dims.keys()) != self.n_modalities:
                    raise AttributeError(
                f"The provided number of input_dims {len(self.input_dims.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )
                encoders = BaseDictEncoders(self.input_dims, model_config.latent_dim)
                self.model_config.uses_default_encoders = True

        if decoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide decoders or input dims for the modalities in the model_config."
                )
            else:
                if len(self.input_dims.keys()) != self.n_modalities:
                    raise AttributeError(
                f"The provided number of input_dims {len(self.input_dims.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )
                decoders = BaseDictDecoders(self.input_dims, model_config.latent_dim)
                self.model_config.uses_default_decoders = True

        self.sanity_check(encoders, decoders)

        self.latent_dim = model_config.latent_dim
        self.model_config = model_config
        self.device = None

        self.set_decoders(decoders)
        self.set_encoders(encoders)

        # Check that the modalities' name are coherent
        if self.input_dims is not None:
            if self.input_dims.keys() != self.encoders.keys():
                print(
                    f"Warning! : The modalities names in model_config.input_dims : {list(self.input_dims.keys())}"
                    f" does not match the modalities names in encoders : {list(self.encoders.keys())}"
                )

        self.use_likelihood_rescaling = model_config.uses_likelihood_rescaling
        if self.use_likelihood_rescaling:
            if self.input_dims is None:
                raise AttributeError(
                    " inputs_dim = None but (use_likelihood_rescaling = True"
                    " in model_config)"
                    " To compute likelihood rescalings we need the input dimensions."
                    " Please provide a valid dictionary for input_dims."
                )
            else:
                max_dim = max(*[np.prod(self.input_dims[k]) for k in self.input_dims])
                self.rescale_factors = {
                    k: max_dim / np.prod(self.input_dims[k]) for k in self.input_dims
                }
        else:
            self.rescale_factors = {k: 1 for k in self.encoders}
            # above, we take the modalities keys in self.encoders as input_dims may be None

        # Set the reconstruction losses
        if model_config.recon_losses is None:
            model_config.recon_losses = {k: "mse" for k in self.encoders}
        self.set_recon_losses(model_config.recon_losses)

    def set_recon_losses(self, recon_dict):
        self.recon_losses = {}  # The loss between reconstruction and true data.
        self.recon_log_probs = (
            {}
        )  # The log probability of true data given the reconstruction
        # recon_log_probs is the normalized negative version of recon_loss and is used for
        # likelihood estimation.

        for k in recon_dict:
            if recon_dict[k] == "mse":
                self.recon_log_probs[k] = lambda input, target: dist.Normal(
                    input, 1
                ).log_prob(target)
                self.recon_losses[k] = MSELoss(reduction="none")
            elif recon_dict[k] == "bce":
                self.recon_log_probs[k] = lambda input, target: dist.Bernoulli(
                    logits=input
                ).log_prob(target)
                self.recon_losses[k] = BCEWithLogitsLoss(reduction="none")
            elif recon_dict[k] == "l1":
                self.recon_log_probs[k] = lambda input, target: dist.Laplace(
                    input, 1
                ).log_prob(target)
                self.recon_losses[k] = L1Loss(reduction="none")
            else:
                raise AttributeError(
                    'Reconstructions losses must be either "mse","bce" or "l1"'
                )
        # TODO : add the possibility to provide custom reconstruction loss and in that case use the negative
        # reconstruction loss as the log probability.

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

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        """
        Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.

        """

        if type(cond_mod) != list and cond_mod != "all":
            raise AttributeError('cond_mod must be either a list or "all"')

        raise NotImplementedError("Must be defined in subclass.")

    def decode(self, embedding: ModelOutput, modalities: Union[list, str] = "all"):
        """Decode a latent variable z in all modalities specified in modalities.

        Args:
            z (ModelOutput): the latent variables. In case there is only one latent space, z is a tensor
                otherwise it is a dictionary containing all the latent variables associated with modalities'name.
            modalities (Union(List, str), Optional): the modalities to decode from z. Default to 'all'.
        Return
            ModelOutput : containing a tensor per modality name.
        """
        self.eval()
        if modalities == "all":
            modalities = list(self.decoders.keys())
        elif type(modalities) == str:
            modalities = [modalities]

        if embedding.one_latent_space:
            z = embedding.z
            outputs = ModelOutput()
            for m in modalities:
                outputs[m] = self.decoders[m](z).reconstruction
            return outputs
        else:
            raise NotImplementedError(
                "The decoding function for multiple latent spaces is not implemented"
                "yet"
            )

    def predict(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        gen_mod: Union[list, str] = "all",
        **kwargs,
    ):
        """Generate in all modalities conditioning on a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to condition on. It must contain at least the modalities
                contained in cond_mod.
            cond_mod (Union[list, str], optional): The modalities to condition on. Defaults to 'all'.
            gen_mod (Union[list, str], optional): The modalities to generate. Defaults to 'all'.

        """
        self.eval()
        N = kwargs.pop("N", 1)
        flatten = kwargs.pop(
            "flatten", False
        )  # If flatten and N>1, the encodings have the shape (Nxn_data, latent_dim)
        # instead of (N, n_data, latent_dim)
        z = self.encode(inputs, cond_mod, N=N, flatten=flatten, **kwargs)
        return self.decode(z, gen_mod)

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
            ``loss = model_output.loss``
        """
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
                cloudpickle.dump(self.encoders, fp)

        if not self.model_config.uses_default_decoders:
            with open(os.path.join(dir_path, "decoders.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoders))
                cloudpickle.dump(self.decoders, fp)

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
    def _load_custom_encoders_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "encoders.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing encoder pkl file ('encoders.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom encoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "encoders.pkl"), "rb") as fp:
                encoder = CPU_Unpickler(fp).load()

        return encoder

    @classmethod
    def _load_custom_decoders_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "decoders.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing decoder pkl file ('decoders.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom decoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "decoders.pkl"), "rb") as fp:
                decoder = CPU_Unpickler(fp).load()

        return decoder

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

        if not model_config.uses_default_encoders:
            encoders = cls._load_custom_encoders_from_folder(dir_path)

        else:
            encoders = None

        if not model_config.uses_default_decoders:
            decoders = cls._load_custom_decoders_from_folder(dir_path)

        else:
            decoders = None

        model = cls(model_config, encoders=encoders, decoders=decoders)
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

            if python_version_minor == "7" and sys.version_info[1] > 7:
                raise LoadError(
                    "Trying to reload a model saved with python3.7 with python3.8+. "
                    "Please create a virtual env with python 3.7 to reload this model."
                )

            elif int(python_version_minor) >= 8 and sys.version_info[1] == 7:
                raise LoadError(
                    "Trying to reload a model saved with python3.8+ with python3.7. "
                    "Please create a virtual env with python 3.8+ to reload this model."
                )
