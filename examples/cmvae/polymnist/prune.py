"""In this file, we execute the pruning procedure for a trained model, to select the optimal number of clusters."""

from multivae.data.datasets.mmnist import MMNISTDataset
from multivae.models import AutoModel

DATA_PATH = "/home/asenella/data"

data = MMNISTDataset(data_path=DATA_PATH, split="train")

model = AutoModel.load_from_hf_hub("asenella/reproduce_cmvae_seed_0", allow_pickle=True)

model.prune_clusters(train_data=data, batch_size=256)
