"""Contains a PolyMNIST classifier and a function to load trained classifiers."""

import torch
from torch import nn


class Flatten(torch.nn.Module):
    """Simple transform to flatten."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class ClassifierPolyMNIST(nn.Module):
    """PolyMNIST classifier.

    Trained classifiers can be downloaded from here: https://zenodo.org/record/4899160/files/PolyMNIST.zip

    .. note ::
        If you are using MultiVae MMNISTDataset and you downloaded the data
        through this class, you have already a `clf` folder in same folder as your data, that was automatically downloaded
        along with the data.

    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),  # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),  # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (980)
            nn.Linear(980, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10),  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        return h


def load_mmnist_classifiers(data_path=".data/clf", device="cpu"):
    """Utility function to load all trained PolyMNISTClassifier for the five modalities.

    If you are using MultiVae MMNISTDataset and you downloaded the data
    through this class, you have a `clf` folder in same folder that was automatically downloaded
    along with the data.

    """
    clfs = {}
    for i in range(5):
        fp = data_path + "/pretrained_img_to_digit_clf_m" + str(i)
        model_clf = ClassifierPolyMNIST()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs
