
from typing import Dict, Any, Union
from torch import device
import torch
from pythae.data.datasets import DatasetOutput

def set_inputs_to_device(inputs: Dict[str, Any], device : str = 'cpu'):
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
                        else:
                            cuda_inputs[key][subkey] = inputs[key][subkey]

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return DatasetOutput(**inputs_on_device)