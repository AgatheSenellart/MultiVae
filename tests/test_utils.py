import pytest
import torch

from multivae.data.utils import set_inputs_to_device
from multivae.models.base.base_utils import rsample_from_gaussian


class Test_set_inputs_to_device:
    """Test the set_inputs_to_device function"""

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
        """Create dictionaries of multimodal data to test the function."""
        return request.param

    def test_function(self, inputs):
        """Check that all the tensors in the input dictionary are set to device."""
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


class Test_rsample_from_gaussian:
    """Test the rsample_from_gaussian function.
    We check that the generated latent sample has the expected shape.
    """

    @pytest.fixture(params=[(5, 10), (10,)])
    def mu_log_var(self, request):
        """Create mean and variance to test the function"""
        return torch.randn(*request.param), torch.randn(*request.param)

    def test(self, mu_log_var):
        """Check the output shape, depending on inputs parameters."""
        mu, lv = mu_log_var

        # test with N=1
        z = rsample_from_gaussian(mu, lv)
        assert z.shape == mu.shape

        z = rsample_from_gaussian(mu, lv, return_mean=True)
        assert torch.all(z == mu)

        # Test with N>1
        z = rsample_from_gaussian(mu, lv, N=10)
        assert z.shape == (10, *mu.shape)

        z = rsample_from_gaussian(mu, lv, N=10, return_mean=True)
        for r in z:
            assert torch.all(r == mu)

        # Test with N>1 and flatten
        z = rsample_from_gaussian(mu, lv, N=10, flatten=True)
        if len(mu.shape) == 1:
            mu_t = mu.unsqueeze(0)
        else:
            mu_t = mu
        assert z.shape == (10 * mu_t.shape[0], mu_t.shape[1])

        z = rsample_from_gaussian(mu, lv, N=10, return_mean=True, flatten=True)
        z_test = z.reshape(10, -1, z.shape[1])
        for r in z_test:
            assert torch.all(r == mu)
