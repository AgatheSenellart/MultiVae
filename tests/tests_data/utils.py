from multivae.data.datasets.base import MultimodalBaseDataset


class test_dataset_plotting(MultimodalBaseDataset):
    """Dataset to test the transform for plotting function"""

    def __init__(self, data, labels=None):
        super().__init__(data, labels)

    def transform_for_plotting(self, tensor, modality):
        return tensor.flatten()
