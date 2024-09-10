from torch import nn
from torch.nn import functional as F, Module
from multivae.models import AutoModel
from torch.utils.data import DataLoader
from multivae.data.datasets import MHD
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data import random_split
import lightning as L
import torchmetrics


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassifierMNIST(L.LightningModule):
    def __init__(self, latent_dim, gmc_model, cond_mod):
        super().__init__()

        self.layer_1 = nn.Linear(latent_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 10)
        
        self.gmc = gmc_model.eval()
        self.loss = CrossEntropyLoss()
        self.cond_mod = cond_mod
        
        self.valid_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)


    def training_step (self, batch):
        
        # set inputs to device
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        labels = batch.labels.to(device)
        
        with torch.no_grad() :
            # encode batch        
            embedding = self.gmc.encode(batch, cond_mod=modality).embedding
            
        # compute logprobs

        x = self.layer_1(embedding)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        probs = self.layer_3(x)
        loss = self.loss(probs, labels)
        self.log('train_loss', loss, logger=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        
        # set inputs to device
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        labels = batch.labels.to(device)
        
        with torch.no_grad() :
            # encode batch        
            embedding = self.gmc.encode(batch, cond_mod=modality).embedding
            
            # compute logprobs

            x = self.layer_1(embedding)
            x = F.relu(x)
            x = self.layer_2(x)
            x = F.relu(x)
            probs = self.layer_3(x)
            eval_loss = self.loss(probs, labels)
            self.log("eval_loss",eval_loss, logger=True, prog_bar=True)
            
            self.valid_accuracy(probs, labels)
            
            return eval_loss
    
    def on_validation_epoch_end(self):
        self.log('valid_acc_epoch', self.valid_accuracy.compute())
        self.valid_accuracy.reset()
    
    def test_step(self, batch):
        
        # set inputs to device
        batch.data = {m : batch.data[m].to(device) for m in batch.data}
        labels = batch.labels.to(device)
        
        with torch.no_grad() :
            # encode batch        
            embedding = self.gmc.encode(batch, cond_mod=modality).embedding
            
            # compute logprobs

            x = self.layer_1(embedding)
            x = F.relu(x)
            x = self.layer_2(x)
            x = F.relu(x)
            probs = self.layer_3(x)
            test_loss = self.loss(probs, labels)
            self.log("test_loss",test_loss)
            
            # Compute accuracy
            self.test_accuracy(probs, labels)
            
            
    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        self.log('test_acc_', test_acc)
        self.test_accuracy.reset()
        return test_acc
            
        
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Load gmc model
dir_path = '/home/asenella/experiments/reproduce_gmc/GMC_training_2024-09-10_14-45-00/final_model'
model = AutoModel.load_from_folder(dir_path)
model = model.to(device)
model = model.eval()


# Define modality for test and train
modality = 'audio'


# define classifier
classifier = ClassifierMNIST(64, model, cond_mod = modality)
classifier = classifier.to(device)


# Define train and eval set
dataset = MHD('/home/asenella/scratch/data/MHD')
test_dataset = MHD('/home/asenella/scratch/data/MHD', split='test')

train_set, val_set = random_split(dataset, [0.9,0.1])

train_data_loader = DataLoader(train_set,batch_size=64)
val_data_loader = DataLoader(val_set, batch_size = 64)
test_data_loader = DataLoader(test_dataset, batch_size = 64)

# checkpoint_callback = L.ModelCheckpoint(
#      monitor='eval_loss',
#      dirpath=f'/home/asenella/experiments/reproduce_gmc/classifier_{modality}',
#      filename='best_model'
#  )

trainer = L.Trainer(
    max_epochs = 50,
    # callbacks = [checkpoint_callback]
)
trainer.fit(classifier, train_data_loader, val_data_loader)

trainer.test(dataloaders=test_data_loader)


# optimizer = Adam(classifier.parameters(), lr=1e-3)

# loss = CrossEntropyLoss()
# for epoch in range(50) :
#     train_loss = 0
#     train_accuracy = 0
#     for i, batch in enumerate(tqdm(train_dataloader)) :
#         optimizer.zero_grad()
#         # set inputs to device
#         batch.data = {m : batch.data[m].to(device) for m in batch.data}
#         labels = batch.labels.to(device)
        
#         with torch.no_grad :
#             # encode batch        
#             embedding = model.encode(batch, cond_mod=modality).embedding
            
#         # compute logprobs
#         probs = classifier(embedding)

#         preds = torch.max(probs, dim=1)[1]
#         epoch_accuracy += (preds == labels).sum()
        
#         output = loss(probs,labels)
#         output.backward()
#         optimizer.step()
        
#         epoch_loss +=  output
        
#     for i, batch in enumerate(tqdm(val_dataloader)) :
        
        
#     print('epoch_loss : ',epoch_loss)
#     print('epoch_accuracy : ',epoch_accuracy/len(train_set))

# # Then compute accuracy on test_set

# test_set  = MHD('/home/asenella/scratch/data/MHD', split='test')
        
# test_loader = DataLoader(test_set)
# accuracy = 0
# for i, batch in enumerate(tqdm(test_loader)):
#     batch.data = {m : batch.data[m].to(device) for m in batch.data}
#     embedding = model.encode(batch, cond_mod=modality).embedding
#     probs = classifier(embedding)
    
#     preds = torch.max(probs, dim=1)[1]
#     labels = batch.labels.to(device)
#     accurate = (preds == labels).sum()
    
#     accuracy = accuracy+accurate
    
    
# accuracy = accuracy/len(test_set)

# print(f'Accuracy on {modality} embedding : {accuracy}')
    
