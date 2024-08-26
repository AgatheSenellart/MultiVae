import torch
import torch.nn as nn

##### Classifiers #####

class Image_Classifier(nn.Module):
    def __init__(self):
        super(Image_Classifier, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)

        return out
    
    
class Sound_Classifier(nn.Module):
    def __init__(self):
        super(Sound_Classifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(2048, 128),
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 64),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(),
                                nn.Linear(64, 10))




    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        return out



class Trajectory_Classifier(nn.Module):
    def __init__(self):
        super(Trajectory_Classifier, self).__init__()

        self.network = nn.Sequential(nn.Linear(200, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 512),
                                  nn.BatchNorm1d(512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 128),
                                  nn.BatchNorm1d(128),
                                  nn.LeakyReLU())
        self.out = nn.Linear(128, 10)


    def forward(self, x):
        h = self.network(x)
        return self.out(h)
    
    


class Label_Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        return x