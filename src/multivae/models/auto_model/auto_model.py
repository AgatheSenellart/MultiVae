import json
import logging
import os

import torch.nn as nn

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AutoModel(nn.Module):
    "Utils class allowing to reload any :class:`multivae.models` automatically"

    def __init__(self) -> None:
        super().__init__()

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

        with open(os.path.join(dir_path, "model_config.json")) as f:
            model_name = json.load(f)["name"]

        if model_name == "JMVAEConfig":
            from ..jmvae import JMVAE

            model = JMVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "JNFConfig":
            from ..jnf import JNF

            model = JNF.load_from_folder(dir_path=dir_path)

        elif model_name == "MMVAEConfig":
            from ..mmvae import MMVAE

            model = MMVAE.load_from_folder(dir_path=dir_path)
        elif model_name == "TELBOConfig":
            from ..telbo import TELBO

            model = TELBO.load_from_folder(dir_path)

        elif model_name == "MVAEConfig":
            from ..mvae import MVAE

            model = MVAE.load_from_folder(dir_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model
