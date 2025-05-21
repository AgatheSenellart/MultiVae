from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class AutoConfig(BaseConfig):
    """Class to reload any multivae.models configuration."""

    @classmethod
    def from_json_file(cls, json_path):
        """Creates a :class:`~multivae.config.BaseMultiVAEConfig` instance from a JSON config file. It
        builds automatically the correct config for any `multivae.models`.

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseMultiVAEConfig`: The created instance
        """
        config_dict = cls._dict_from_json(json_path)
        config_name = config_dict.pop("name")

        if config_name == "BaseMultiVAEConfig":
            from ..base import BaseMultiVAEConfig

            model_config = BaseMultiVAEConfig.from_json_file(json_path)

        elif config_name == "JMVAEConfig":
            from ..jmvae import JMVAEConfig

            model_config = JMVAEConfig.from_json_file(json_path)

        elif config_name == "JNFConfig":
            from ..jnf import JNFConfig

            model_config = JNFConfig.from_json_file(json_path)

        elif config_name == "MMVAEConfig":
            from ..mmvae import MMVAEConfig

            model_config = MMVAEConfig.from_json_file(json_path)
        elif config_name == "TELBOConfig":
            from ..telbo import TELBOConfig

            model_config = TELBOConfig.from_json_file(json_path)

        elif config_name == "MVAEConfig":
            from ..mvae import MVAEConfig

            model_config = MVAEConfig.from_json_file(json_path)

        elif config_name == "MoPoEConfig":
            from ..mopoe import MoPoEConfig

            model_config = MoPoEConfig.from_json_file(json_path)

        elif config_name == "MVTCAEConfig":
            from ..mvtcae import MVTCAEConfig

            model_config = MVTCAEConfig.from_json_file(json_path)

        elif config_name == "MMVAEPlusConfig":
            from ..mmvaePlus import MMVAEPlusConfig

            model_config = MMVAEPlusConfig.from_json_file(json_path)

        elif config_name == "NexusConfig":
            from ..nexus import NexusConfig

            model_config = NexusConfig.from_json_file(json_path)

        elif config_name == "CVAEConfig":
            from ..cvae import CVAEConfig

            model_config = CVAEConfig.from_json_file(json_path)

        elif config_name == "MHVAEConfig":
            from ..mhvae import MHVAEConfig

            model_config = MHVAEConfig.from_json_file(json_path)

        elif config_name == "DMVAEConfig":
            from ..dmvae import DMVAEConfig

            model_config = DMVAEConfig.from_json_file(json_path)

        elif config_name == "CMVAEConfig":
            from ..cmvae import CMVAEConfig

            model_config = CMVAEConfig.from_json_file(json_path)

        elif config_name == "CRMVAEConfig":
            from ..crmvae import CRMVAEConfig

            model_config = CRMVAEConfig.from_json_file(json_path)

        else:
            raise NameError(
                "Cannot reload automatically the model configuration... "
                f"The model name in the `model_config.json may be corrupted. Got `{config_name}`"
            )

        return model_config
