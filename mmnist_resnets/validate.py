

from global_config import *
from multivae.models import AutoModel
from multivae.data.datasets import MMNISTDataset
import time
from multivae.trainers.base.callbacks import load_wandb_path_from_folder

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 0
dir = '/home/asenella/experiments/mmnist_resnets/MoPoE/seed_0/MoPoE_training_2024-09-20_17-33-05/final_model'
wandb_path = load_wandb_path_from_folder(dir)

model = AutoModel.load_from_folder(dir)
model = model.eval().to(device)
model.device = device

t1 = time.time()
eval_model(model, dir,train_data=train_data, test_data=test_data,wandb_path=wandb_path,seed=seed)
t2 = time.time()
print(t2-t1)