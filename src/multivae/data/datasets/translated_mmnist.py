import argparse
import glob
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from multivae.data.datasets import MultimodalBaseDataset
from multivae.data.datasets.mmnist import DatasetOutput

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class TranslatedMMNIST(MultimodalBaseDataset):  # pragma: no cover
    """
    Translated version of the PolyMNIST dataset.
    The data is built from background images that need to be downloaded beforehand.

    The original PolyMNIST (5 modalities) background images can be downloaded from : https://mybox.inria.fr/d/78e581ee5b07402983fa/.

    To use the ExtendedPolyMNIST dataset (10 modalities) introduced in "Score-Based Multimodal Autoencoder" (Wesego 2024),
    download the background images from https://github.com/rooshenasgroup/sbmae/tree/main/poly_background.



    .. code-block:: python

        >>> from multivae.data.datasets import TranslatedMMNIST
        >>> dataset = TranslatedMMNIST(
        ...   path = 'your_data_path',
        ...   scale = 0.75, # downscale 75%
        ...   translate = True, # random translation
        ...   n_modalities = 5,
        ...   background_path = 'path_to_background_image'
        ...)

    Args:
            path (str): parent path where to save the dataset
            scale (float): The scale factor to downsample the MNIST images
            translate (bool): Wether to translate the MNIST images
            n_modalities (int): The number of modalities. It must match the number of background images.
            background_path (str, optional): Path to the background images. Defaults to None.
            split (str, optional): train or test. Defaults to 'train'.
            transform (torchvision.transforms, optional): The transform to apply to images. Defaults to ToTensor().
            target_transform (torchvision.transform, optional): The transform to apply to labels. Defaults to None.

    """

    def __init__(
        self,
        path: str,
        scale: float,
        translate: bool,
        n_modalities: int,
        background_path=None,
        split="train",
        transform=ToTensor(),
        target_transform=None,
    ):

        self.scale = scale
        self.translate = translate
        self.parent_path = path
        self.save_path = os.path.join(
            path,
            f"Translated_MMNIST_scale_{int(scale*100)}_translated_{translate}",
            split,
        )

        self.num_modalities = n_modalities
        unimodal_datapaths = [
            os.path.join(self.save_path, f"m{i}") for i in range(self.num_modalities)
        ]
        self.transform = transform
        self.target_transform = target_transform

        self.check_or_create_dataset(unimodal_datapaths, background_path, split)

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in unimodal_datapaths}
        for dp in unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

    def check_or_create_dataset(self, unimodal_paths, background_path, split):
        """Check if the dataset exists at the provided path and if not creates the dataset from the background images"""

        data_exists = True
        for p in unimodal_paths:
            data_exists = os.path.exists(p)
        if not data_exists:
            if background_path is None:
                raise ValueError(
                    "The provided path does not contain the dataset in the proper format"
                    " and no background path was provided."
                )
            if not os.path.exists(background_path):
                raise ValueError(f"Provided path {background_path} doesn't exist")

            logger.info("Dataset not found, creating dataset from the background path.")
            self._create_mmnist_dataset(background_path, split == "train")

    def _create_mmnist_dataset(self, background_path, train: bool):
        """Created the Multimodal MNIST Dataset under 'savepath' given a directory of background images.

        Args:
            savepath (str): path to directory that the dataset will be written to.
                This path must also contain a 'background' folder where the background images are stored.
            num_modalities (int): number of modalities to create.
            train (bool): create the dataset based on MNIST training (True) or test data (False).
            translate_mnist (bool): downsample MNIST by a factor of 2 and place it at a random x/y-coordinate

        """

        # load MNIST data
        mnist = datasets.MNIST(
            self.parent_path, train=train, download=True, transform=None
        )

        # load background images
        background_filepaths = sorted(glob.glob(os.path.join(background_path, "*.jpg")))

        logger.info("\nbackground_filepaths:\n" + str(background_filepaths) + "\n")
        if self.num_modalities > len(background_filepaths):
            raise ValueError(
                "Number of background images must be larger or equal to number of modalities"
            )
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(self.num_modalities):
            unimodal_path = os.path.join(self.save_path, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(self.num_modalities):
                ixs_perm = ixs[
                    torch.randperm(len(ixs))
                ]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = self._add_background_image(
                        background_images[m], mnist.data[ix]
                    )
                    # save as png
                    filepath = os.path.join(
                        self.save_path, "m%d/%d.%d.png" % (m, i, digit)
                    )
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        logger.info(
                            "Saved %d/%d images to %s"
                            % (cnt, len(mnist) * self.num_modalities, self.save_path)
                        )
        assert cnt == len(mnist) * self.num_modalities

    def _add_background_image(
        self, background_image_pil, mnist_image_tensor, change_colors=False
    ):

        # translate mnist image: downsamle digit and place it at a random location
        if self.translate is True:
            mnist_image_tensor_downsampled = torch.nn.functional.interpolate(
                mnist_image_tensor.unsqueeze(0).float(),
                scale_factor=self.scale,
                mode="bilinear",
            )
            mnist_image_tensor = mnist_image_tensor * 0  # black out everything

            x = np.random.randint(0, int(28 * (1 - self.scale)))
            y = np.random.randint(0, int(28 * (1 - self.scale)))
            mnist_image_tensor[
                :, x : x + int(28 * (self.scale)), y : y + int(28 * (self.scale))
            ] = mnist_image_tensor_downsampled

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(torch.bool)

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError(
                "Unexpected dimensionality of MNIST image:", img_binarized.shape
            )

        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = [self.file_paths[dp][index] for dp in self.file_paths]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]
        labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]

        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]
        if self.target_transform:
            labels = [self.transform(label) for label in labels]

        images_dict = {f"m{m}": images[m] for m in range(self.num_modalities)}
        return DatasetOutput(data=images_dict, labels=labels[0])

    def __len__(self):
        return self.num_files
