from typing import Any, Dict

import torch
from pythae.data.datasets import DatasetOutput


def set_inputs_to_device(inputs: Dict[str, Any], device: str = "cpu"):
    """Set an dict input to device. It covers the case where the input is a 
    Dict[str, tensor], Dict[str, dict[str, tensor]], Dict[str, dict[str, dict[str, Tensor]]]"""

    inputs_on_device = inputs

    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)

        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()

            elif isinstance(inputs[key], dict):
                cuda_inputs[key] = dict.fromkeys(inputs[key])
                for subkey in inputs[key].keys():
                    if torch.is_tensor(inputs[key][subkey]):
                        cuda_inputs[key][subkey] = inputs[key][subkey].cuda()
                    elif isinstance(inputs[key][subkey], dict):
                        cuda_inputs[key][subkey] = dict.fromkeys(inputs[key][subkey])
                        for subsubkey in inputs[key][subkey].keys():
                            if torch.is_tensor(inputs[key][subkey][subsubkey]):
                                cuda_inputs[key][subkey][subsubkey] = inputs[key][
                                    subkey
                                ][subsubkey].cuda()
                            else:
                                cuda_inputs[key][subkey][subsubkey] = inputs[key][
                                    subkey
                                ][subsubkey]
                    else:
                        cuda_inputs[key][subkey] = inputs[key][subkey]
            else:
                cuda_inputs[key] = inputs[key]
        inputs_on_device = cuda_inputs

    return DatasetOutput(**inputs_on_device)


def get_batch_size(inputs: Dict[str, Any]):
    """Get the batch size from an batch input"""
    k = list(inputs.data.keys())[0]
    return len(inputs.data[k])


def drop_unused_modalities(inputs: Dict[str, Any]):
    """Drops modalities that are unavailable for an entire batch."""
    if not hasattr(inputs, "masks"):
        return inputs
    else:
        mods = list(inputs.masks.keys()).copy()
        for m in mods:
            if not torch.any(inputs.masks[m]):
                inputs.data.pop(m)
                inputs.masks.pop(m)
        return inputs


class MinMaxScaler(torch.nn.Module):
    """Transforms each modality inputs so that it has values between 0 and 1."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_fitted = False

    def fit(self, embeddings: Dict[str, torch.tensor]):
        self.mins = {m: torch.min(embeddings[m]).detach() for m in embeddings}
        self.maxs = {m: torch.max(embeddings[m]).detach() for m in embeddings}

        self.is_fitted = True

    def forward(self, embeddings: Dict[str, torch.tensor]):
        new_embeddings = dict()
        for m in embeddings:
            new_embeddings[m] = (embeddings[m] - self.mins[m]) / (
                self.maxs[m] - self.mins[m]
            )
        return new_embeddings

    def forward_modality(self, modality_embedding: torch.tensor, modality: str):
        return (modality_embedding - self.mins[modality]) / (
            self.maxs[modality] - self.mins[modality]
        )
