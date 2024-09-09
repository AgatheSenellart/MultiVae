from torch import nn
from torch.nn import functional as F, Module
from multivae.models import AutoModel
from torch.utils.data import DataLoader
from multivae.data.datasets import MHD
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch

class ClassifierMNIST(Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.layer_1 = nn.Linear(latent_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x

classifier = ClassifierMNIST(64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = classifier.to(device)

dir_path = '/home/asenella/experiments/reproduce_gmc/GMC_training_2024-09-06_20-19-09/final_model'
model = AutoModel.load_from_folder(dir_path)

model = model.to(device)

modality = 'audio'

train_set = MHD('/home/asenella/scratch/data/MHD')

data_loader = DataLoader(train_set,batch_size=64)

optimizer = Adam(classifier.parameters(), lr=1e-3)

loss = CrossEntropyLoss()
for epoch in range(50) :
    epoch_loss = 0
    for i, batch in enumerate(tqdm(data_loader)) :
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        optimizer.zero_grad()
        labels = batch.labels.to(device)
        embedding = model.encode(batch, cond_mod=modality).embedding
        
        probs = classifier(embedding)
        
        output = loss(probs,labels)
        output.backward()
        optimizer.step()
        
        epoch_loss +=  output
    print('epoch_loss : ',epoch_loss)

# Then compute accuracy on test_set

test_set  = MHD('/home/asenella/scratch/data/MHD', split='test')
        
test_loader = DataLoader(test_set)
accuracy = 0
for i, batch in enumerate(tqdm(test_loader)):
    batch.data = {m : batch.data[m].to(device) for m in batch.data}
    embedding = model.encode(batch, cond_mod=modality).embedding
    probs = classifier(embedding)
    
    preds = torch.max(probs, dim=1)[1]
    labels = labels.to(device)
    accurate = (preds == labels).sum()
    
    accuracy = accuracy+accurate
    
    print('batch_accuracy', accurate/len(preds))
    
accuracy = accuracy/len(test_set)

print(f'Accuracy on {modality} embedding : {accuracy}')
    
