from config2 import *

from multivae.models import AutoModel
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from torch.utils.data import DataLoader
from multivae.data.utils import set_inputs_to_device



train_data = MMNISTDataset(
    data_path= '/Users/agathe/dev/data',
    split="train",
    missing_ratio=0,
    keep_incomplete=True,
)

test_data = MMNISTDataset(data_path='/Users/agathe/dev/data', split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(0)
)

device = 'cpu'

model = AutoModel.load_from_folder('JNFDcca_training_2023-07-11_13-24-45/final_model')
model.to(device)

#### Compute all embeddings for the DCCA ####

test_loader = DataLoader(test_data,512,)

embeddings = {k : [] for k in model.dcca_networks}
labels = []
for batch in test_loader:
    
    batch = set_inputs_to_device(batch , device=device)
    
    for m in batch.data:
        enc = model.dcca_networks[m](batch.data[m])
        embeddings[m].append(enc)
        
    labels.append(batch.labels)
        

embeddings = {k : torch.cat(embeddings[k], dim=0) for k in embeddings}
labels = torch.cat(labels, dim=0)




    



import pickle
with open('embeddings.pkl', 'w+') as f:
    pickle.dump(embeddings, f)
    

    