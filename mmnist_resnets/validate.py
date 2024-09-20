

from config2 import *
from multivae.models import AutoModel
from multivae.data.datasets import MMNISTDataset
import time

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

dir = '/home/asenella/dev/multivae_package/mmnist_resnet/JNF/seed_0/JNF_training_2024-09-03_15-02-21/final_model'

model = AutoModel.load_from_folder(dir)
wandb_path = 'multimodal_vaes/mmnist_resnet/fco6kczk'

model = model.eval().cuda()


t1 = time.time()
eval_model(model, dir,train_data=train_data, test_data=test_data,wandb_path=wandb_path,seed=0)
t2 = time.time()
print(t2-t1)