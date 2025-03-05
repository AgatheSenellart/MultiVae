import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        s0 = self.s0 = 7
        nf = self.nf = 64
        nf_max = self.nf_max = 1024
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 10)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(actvn(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
    
def load_classifiers(data_path, device="cpu"):
    clfs = {}
    for i in range(5):
        fp = data_path + f"/m{i}/classifier.pt" 
        model_clf = DigitClassifier()
        model_clf.load_state_dict(torch.load(fp, map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs

    
    
if __name__ == "__main__" :
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", type=int)
    args = parser.parse_args()
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision.datasets import DatasetFolder
    from dataset import MMNISTDataset
    import torch.optim as optim
    from torchvision.transforms import ToTensor
    from tqdm import tqdm
    import os
    
    mod = f'm{args.mod}'
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = f'/home/asenella/scratch/data/translated_mmnist_2/classifiers/m{args.mod}'
    
    # Define classifier :
    
    classifier = DigitClassifier().to(device)
    
    unimodal_paths = ['/home/asenella/scratch/data/translated_mmnist_2/train/m'+str(i) for i in range(5)]

    trainset = MMNISTDataset(unimodal_paths, transform=ToTensor() )
    
    loss = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(trainset,batch_size=256)
    
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    import logging
    # Create a logger object
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Create a file handler to log messages to a file
    handler = logging.FileHandler(output_path + '/training.log')
    handler.setLevel(logging.DEBUG)
    
    # Define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Attach the handler to the logger
    logger.addHandler(handler)

    # Log some messages
    logger.debug('This is a debug message')
    logger.info('This is an info message')

    
    for i in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            data = batch.data[mod].to(device)
            labels = batch.labels.to(device)
            
            outputs = classifier(data)
            
            loss_batch = loss(outputs, labels)
            
            loss_batch.backward()
            
            optimizer.step()
            
            epoch_loss+= loss_batch.item()
            
        logger.info('Epoch loss :', epoch_loss)
        
    logger.info('Finished training.')
    
    # Test
    
    unimodal_paths = ['/home/asenella/scratch/data/translated_mmnist_2/test/m'+str(i) for i in range(5)]

    testset = MMNISTDataset(unimodal_paths, transform=ToTensor() )
        
    test_loader = DataLoader(testset,32)
    total=0
    correct = 0
    for batch in test_loader:
            
           
            data = batch.data[mod].to(device)
            labels = batch.labels.to(device)
            
            outputs = classifier(data)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                
                
    logger.info('Test accuracy :', correct/total)
    
    # Save the model
    torch.save(classifier.state_dict(),os.path.join(output_path, 'classifier.pt'))

            