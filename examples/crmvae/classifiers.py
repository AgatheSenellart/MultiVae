"""Define and train classifiers for the translated PolyMNIST"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def actvn(x):
    """Custom activation function"""
    out = F.leaky_relu(x, 2e-1)
    return out


class DigitClassifier(nn.Module):
    """Resnet Classifier for the Translated PolyMNIST dataset"""

    def __init__(self):
        super().__init__()
        s0 = self.s0 = 7
        nf = self.nf = 64
        nf_max = self.nf_max = 1024
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def load_classifiers(data_path, device="cpu"):
    """Function to load all pretrained classifiers."""
    clfs = {}
    for i in range(5):
        fp = data_path + f"/m{i}/classifier.pt"
        model_clf = DigitClassifier()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs[f"m{i}"] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError(f"Classifier is 'None' for modality {str(i)}")
    return clfs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int)  # The modality to train
    args = parser.parse_args()

    import os

    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from multivae.data.datasets import TranslatedMMNIST

    DATA_PATH = "/scratch/asenella/data"
    MMNIST_BACKGROUND_PATH = DATA_PATH + "/mmnist_background"
    MOD = f"m{args.mod}"
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_PATH = (
        f"/home/asenella/scratch/test/translated_mmnist_2/classifiers/m{args.mod}"
    )

    # Define classifier :

    classifier = DigitClassifier().to(DEVICE)

    trainset = TranslatedMMNIST(
        DATA_PATH,
        scale=0.75,
        translate=True,
        n_modalities=5,
        background_path=MMNIST_BACKGROUND_PATH,
    )

    loss = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(trainset, batch_size=256)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    for i in tqdm(range(NUM_EPOCHS)):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            data = batch.data[MOD].to(DEVICE)
            labels = batch.labels.to(DEVICE)

            outputs = classifier(data)

            loss_batch = loss(outputs, labels)

            loss_batch.backward()

            optimizer.step()

            epoch_loss += loss_batch.item()

        print("Epoch loss :%s", epoch_loss)

    print("Finished training.")

    # Test

    testset = TranslatedMMNIST(
        SAVE_PATH,
        scale=0.75,
        translate=True,
        n_modalities=5,
        background_path=MMNIST_BACKGROUND_PATH,
        split="test",
    )

    test_loader = DataLoader(testset, 32)
    total = 0
    correct = 0
    for batch in test_loader:
        data = batch.data[MOD].to(DEVICE)
        labels = batch.labels.to(DEVICE)

        outputs = classifier(data)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Test accuracy :%s", correct / total)

    # Save the model
    torch.save(classifier.state_dict(), os.path.join(SAVE_PATH, "classifier.pt"))
