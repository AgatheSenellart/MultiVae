from config2 import *

from multivae.models import AutoModel
from multivae.trainers import AddDccaTrainer, AddDccaTrainerConfig
from multivae.models.nn.default_architectures import MultipleHeadJointEncoder
from multivae.models.nn.mmnist import DecoderConvMMNIST, EncoderConvMMNIST_adapted
from torch.utils.data import DataLoader



train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train",
    missing_ratio=0,
    keep_incomplete=True,
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")

train_data, eval_data = random_split(
    train_data, [0.9, 0.1], generator=torch.Generator().manual_seed(0)
)


model = AutoModel.load_from_folder('/home/asenella/dev/multivae_package/compare_on_mmnist/config_resnet/JNFDcca/seed_0/missing_ratio_0/JNFDcca_training_2023-07-11_13-24-45/final_model')

#### Compute all embeddings for the DCCA ####

test_loader = DataLoader(test_data,512,)