import json
import logging
import os

from torch import nn

from ..base.base_utils import hf_hub_is_available

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class AutoModel(nn.Module):
    """Utils class allowing to reload any :class:`multivae.models` automatically."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def load_from_folder(cls, dir_path: str):
        """Class method to be used to load the model from a specific folder.

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

        elif model_name == "MoPoEConfig":
            from ..mopoe import MoPoE

            model = MoPoE.load_from_folder(dir_path)

        elif model_name == "MVTCAEConfig":
            from ..mvtcae import MVTCAE

            model = MVTCAE.load_from_folder(dir_path)

        elif model_name == "MMVAEPlusConfig":
            from ..mmvaePlus import MMVAEPlus

            model = MMVAEPlus.load_from_folder(dir_path)

        elif model_name == "NexusConfig":
            from ..nexus import Nexus

            model = Nexus.load_from_folder(dir_path)

        elif model_name == "CVAEConfig":
            from ..cvae import CVAE

            model = CVAE.load_from_folder(dir_path)

        elif model_name == "MHVAEConfig":
            from ..mhvae import MHVAE

            model = MHVAE.load_from_folder(dir_path)

        elif model_name == "DMVAEConfig":
            from ..dmvae import DMVAE

            model = DMVAE.load_from_folder(dir_path)

        elif model_name == "CMVAEConfig":
            from ..cmvae import CMVAE

            model = CMVAE.load_from_folder(dir_path)

        elif model_name == "CRMVAEConfig":
            from ..crmvae import CRMVAE

            model = CRMVAE.load_from_folder(dir_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model

    @classmethod
    def load_from_hf_hub(
        cls, hf_hub_path: str, allow_pickle: bool = False
    ):  # pragma: no cover
        """Class method to be used to load a automaticaly a pretrained model from the Hugging Face
        hub.

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        logger.info("Downloading config file ...")

        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        with open(os.path.join(dir_path, "model_config.json")) as f:
            model_name = json.load(f)["name"]

        if model_name == "JMVAEConfig":
            from ..jmvae import JMVAE

            model = JMVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "JNFConfig":
            from ..jnf import JNF

            model = JNF.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MMVAEConfig":
            from ..mmvae import MMVAE

            model = MMVAE.load_from_hf_hub(hf_hub_path, allow_pickle)
        elif model_name == "TELBOConfig":
            from ..telbo import TELBO

            model = TELBO.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MVAEConfig":
            from ..mvae import MVAE

            model = MVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MoPoEConfig":
            from ..mopoe import MoPoE

            model = MoPoE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MVTCAEConfig":
            from ..mvtcae import MVTCAE

            model = MVTCAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MMVAEPlusConfig":
            from ..mmvaePlus import MMVAEPlus

            model = MMVAEPlus.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "NexusConfig":
            from ..nexus import Nexus

            model = Nexus.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "CVAEConfig":
            from ..cvae import CVAE

            model = CVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "DMVAEConfig":
            from ..dmvae import DMVAE

            model = DMVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "CMVAEConfig":
            from ..cmvae import CMVAE

            model = CMVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "MHVAEConfig":
            from ..mhvae import MHVAE

            model = MHVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        elif model_name == "CRMVAEConfig":
            from ..crmvae import CRMVAE

            model = CRMVAE.load_from_hf_hub(hf_hub_path, allow_pickle)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model
