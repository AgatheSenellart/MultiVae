"""
Multimodal dataset wrapper for the MNIST labels dataset.
"""

import io
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch.distributions import Bernoulli
from torchvision.datasets import MNIST

from .base import DatasetOutput, MultimodalBaseDataset


class MnistLabels(MultimodalBaseDataset):  # pragma: no cover
    """Mnist-Labels dataset. The first modality is the image and the second modality is the label."""

    def __init__(
        self,
        data_path: str,
        split: Literal["train", "test"] = "train",
        download=False,
        random_binarized=True,
        dtype=torch.float32,
    ):
        """
        Class to wrap the MnistLabels dataset for MultiVae use.

        Args:
            data_path (str): Where is stored the data.
            split (Literal["train", "test"], optional): The split to use. Defaults to "train".
            download (bool, optional): If the data is not available at the given data_path, whether to download it or not.
                Defaults to False.
            random_binarized (bool, optional): Whether binarize the images using a Bernoulli distribution for each pixel.
                Defaults to True.
            dtype (_type_, optional): Type for the arrays. Defaults to torch.float32.
        """
        torchvision_dataset = MNIST(
            root=data_path, train=(split == "train"), download=download
        )

        self.images = torchvision_dataset.data.div(255).unsqueeze(1).to(dtype)
        self.labels = torchvision_dataset.targets
        self.labels_one_hot = torch.zeros(len(self.labels), 10)
        self.labels_one_hot = self.labels_one_hot.scatter(
            1, self.labels.unsqueeze(1), 1
        )
        self.labels_one_hot = self.labels_one_hot.unsqueeze(1)
        self.random_binarized = random_binarized

    def __getitem__(self, index):
        if self.random_binarized:
            images = Bernoulli(self.images[index]).sample()
        else:
            images = self.images[index]
        return DatasetOutput(
            data=dict(images=images, labels=self.labels_one_hot[index]),
            labels=self.labels[index],
        )

    def __len__(self):
        return len(self.labels)

    def transform_for_plotting(self, tensor, modality):
        """Transforms the label modality to text for plotting."""
        if modality == "labels":
            list_images = []
            tensor = torch.argmax(tensor, dim=-1)
            for t in tensor:
                list_images.append(self.to_text(t))

            return torch.stack(list_images)

        return tensor

    def to_text(self, int_label):

        device = int_label.device

        fig = plt.figure(figsize=(0.2, 0.2))
        plt.text(
            x=0.5,
            y=0.5,
            s=str(int_label.item()),
            fontsize=7,
            verticalalignment="center_baseline",
            horizontalalignment="center",
        )
        plt.axis("off")
        fig.tight_layout()
        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        image = PIL.Image.open(img_buf)

        image = np.array(image).transpose(2, 0, 1) / 255
        plt.close(fig=fig)
        return torch.from_numpy(image).float().to(device)
