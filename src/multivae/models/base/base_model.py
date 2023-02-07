import torch
import torch.nn as nn

from .base_config import BaseMultiVAEConfig

class BaseMultiVAE(nn.Module):
    def __init__(self, model_config: BaseMultiVAEConfig=None) -> None:
        super().__init__()

        self.dummy_param = nn.Parameter(torch.tensor([1.]))

        self.model_name = "BaseMultiVAE"
        self.model_config = model_config

    def forward(self, inputs):
        pass