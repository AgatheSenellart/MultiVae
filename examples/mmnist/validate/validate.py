from multivae.models.auto_model import AutoConfig, AutoModel
from multivae.data.datasets import MMNISTDataset
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import torch 
from tqdm import tqdm

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),     # -> (10, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),    # -> (20, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),                                                # -> (980)
            nn.Linear(980, 128),                                      # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)                                        # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        # return F.log_softmax(h, dim=-1)
        return h
    
    

def load_mmnist_classifiers(data_path =  "../../../data/clf",device='cuda'):

    clfs = {}
    for i in range(5):
        fp = data_path + '/pretrained_img_to_digit_clf_m' + str(i)
        model_clf = ClfImg()
        model_clf.load_state_dict(torch.load(fp,map_location=torch.device(device)))
        model_clf = model_clf.to(device)
        clfs["m%d" % i] = model_clf
    for m, clf in clfs.items():
        if clf is None:
            raise ValueError("Classifier is 'None' for modality %s" % str(i))
    return clfs


##############################################################################

test_set = MMNISTDataset(data_path = "../../../data/MMNIST",split="test")
test_loader = DataLoader(test_set, batch_size=512)

data_path = 'dummy_output_dir/JNF_training_2023-03-15_20-20-36/final_model'

f= open(data_path + '/metrics.txt',"w+")


model = AutoModel.load_from_folder(data_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.load_from_folder(data_path).to(device)


# Load classifiers
clfs = load_mmnist_classifiers(device=device)

def pair_accuracies(model,test_loader, clfs,f,device):
    accuracies = {}
    for batch in tqdm(test_loader):
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        batch.labels = batch.labels.to(device)
        for cond_m in model.encoders:
            for pred_m in model.encoders:
                output = model.predict(batch, cond_m, pred_m)
                preds = clfs[pred_m](output[pred_m])
                pred_labels = torch.argmax(preds,dim=1)
                try :
                    accuracies[cond_m + '_' + pred_m] += torch.sum(pred_labels == batch.labels)
                except:
                    accuracies[cond_m + '_' + pred_m] = torch.sum(pred_labels == batch.labels)
                    
    acc = {k: accuracies[k].cpu().numpy()/len(test_set) for k in accuracies}
    f.write('Pair accuracies \n')
    mean_pair_acc = np.mean(list(acc.values()))
    f.write(acc.__str__() + '\n')
    f.write('Mean pair accuracies' + str(mean_pair_acc))
    return acc


def all_one_accuracies(model,test_loader, clfs,f,device):
    accuracies = {}
    for batch in tqdm(test_loader):
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        batch.labels = batch.labels.to(device)
        for pred_m in model.encoders:
            cond_m = [m for m in model.encoders if m != pred_m]
            output = model.predict(batch, cond_m, pred_m)
            preds = clfs[pred_m](output[pred_m])
            pred_labels = torch.argmax(preds,dim=1)
            try :
                accuracies[pred_m] += torch.sum(pred_labels == batch.labels)
            except:
                accuracies[pred_m] = torch.sum(pred_labels == batch.labels)

    acc = {k: accuracies[k].cpu().numpy()/len(test_set) for k in accuracies}
    f.write('All to one accuracies \n')
    mean_pair_acc = np.mean(list(acc.values()))
    f.write(acc.__str__() + '\n')
    f.write('Mean all-to-one accuracies' + str(mean_pair_acc))
    return acc


pair_accuracies(model,test_loader,clfs,f,device)
all_one_accuracies(model,test_loader,clfs,f,device)

# ll = 0
# nb_batch =0
# for batch in test_loader:
#     ll += model.compute_joint_nll(batch)
#     nb_batch+=1
# print(ll/nb_batch)