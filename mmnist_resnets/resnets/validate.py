

from config2 import *
from multivae.models import AutoModel
from multivae.data.datasets import MMNISTDataset

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

dir = '/home/asenella/dev/multivae_package/mmnist_resnet/MMVAEPlus/K_1/seed_0/MMVAEPlus_training_2024-09-03_15-02-21/final_model'

model = AutoModel.load_from_folder(dir)
wandb_path = 'multimodal_vaes/mmnist_resnet/y6a57slw'

model = model.eval().cuda()



eval_model(model, dir,train_data=train_data, test_data=test_data,wandb_path=wandb_path,seed=0)