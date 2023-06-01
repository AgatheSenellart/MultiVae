import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN
from torchvision.transforms import ToTensor


class SVHN_Classifier(nn.Module):
    def __init__(self):
        super(SVHN_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_mnist_svhn_classifiers(data_path, device="cuda"):
    c1 = MNIST_Classifier()
    c1.load_state_dict(torch.load(f"{data_path}/mnist.pt", map_location=device))
    c2 = SVHN_Classifier()
    c2.load_state_dict(torch.load(f"{data_path}/svhn.pt", map_location=device))
    return {"mnist": c1.to(device).eval(), "svhn": c2.to(device).eval()}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 30
    train_set = {
        "mnist": MNIST("../data", transform=ToTensor()),
        "svhn": SVHN("../data", transform=ToTensor()),
    }
    test_set = {
        "mnist": MNIST("../data", train=False, transform=ToTensor()),
        "svhn": SVHN("../data", split="test", transform=ToTensor()),
    }
    classifiers = {"mnist": MNIST_Classifier(), "svhn": SVHN_Classifier()}

    for modality in ["mnist", "svhn"]:
        train_loader = DataLoader(train_set[modality], batch_size=128)
        classifier = classifiers[modality].to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            total_iters = len(train_loader)
            print("\n====> Epoch: {:03d} ".format(epoch))
            for i, data in enumerate(train_loader):
                # get the inputs
                x, targets = data
                x, targets = x.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = classifier(x)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (i + 1) % 5 == 0:
                    print(
                        "iteration {:04d}/{:d}: loss: {:6.3f}".format(
                            i + 1, total_iters, running_loss / 1000
                        )
                    )
                    running_loss = 0.0
        print("Finished Training, calculating test loss...")

        classifier.eval()
        total = 0
        correct = 0
        test_loader = DataLoader(test_set[modality], batch_size=128)

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, targets = data
                x, targets = x.to(device), targets.to(device)
                outputs = classifier(x)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(
            "The classifier correctly classified {} out of {} examples. Accuracy: "
            "{:.2f}%".format(correct, total, correct / total * 100)
        )
        if not os.path.exists("../../classifiers"):
            os.mkdir("../../classifiers")
        torch.save(classifier.state_dict(), f"../../classifiers/{modality}.pt")
