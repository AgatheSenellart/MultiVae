import pytest
from pythae.data.datasets import DatasetOutput
from torchvision.datasets import MNIST

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mnist_svhn import MnistSvhn


@pytest.mark.slow
class TestMNISTSVHN:
    """Test class for MNISTSVHN dataset.
    This test only works locally with the MNISTSVHN downloaded in the ../data folder.
    """

    @pytest.fixture
    def input_dataset_test(self):
        """Create the datafolder."""
        data_path = "../data"
        split = "test"

        return dict(data_path=data_path, split=split)

    def test_create_dataset(self, input_dataset_test):
        """Test the MnistSVHN dataset.
        We check the output and lenght of the dataset.
        """
        try:
            MNIST(
                input_dataset_test["data_path"],
                train=(input_dataset_test["split"] == "train"),
                download=False,
            )
        except RuntimeError:  # If the dataset is not available don't run the test
            print(
                "The dataset is not found at ../data/"
                "The test on MnistSvhn dataset will not run."
            )
            return

        dataset = MnistSvhn(**input_dataset_test)
        assert isinstance(dataset, MultimodalBaseDataset)
        sample = dataset[0]
        assert isinstance(sample, DatasetOutput)
        assert len(dataset) == 50000
