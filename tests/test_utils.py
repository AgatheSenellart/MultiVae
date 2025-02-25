import pytest
import torch

from multivae.data.utils import set_inputs_to_device


class Test_set_inputs_to_device:

    @pytest.fixture(
        params=[
            dict(image=torch.randn(10), image2=torch.randn(10)),
            dict(
                image=torch.randn(10),
                text=dict(one_hot=torch.ones(10), tokens=torch.tensor([1, 2.0])),
            ),
        ]
    )
    def inputs(self, request):
        return request.param

    def test(self, inputs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs_on_device = set_inputs_to_device(inputs, device)

        assert isinstance(inputs_on_device, dict)
        for key in inputs_on_device:
            if isinstance(inputs_on_device[key], torch.Tensor):
                assert inputs_on_device[key].is_cuda == (device == "cuda")
            else:
                for subkey in inputs_on_device[key]:
                    if isinstance(inputs_on_device[key][subkey], torch.Tensor):
                        assert inputs_on_device[key][subkey].is_cuda == (
                            device == "cuda"
                        )
