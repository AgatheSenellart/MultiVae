from dataset import MMNISTDataset

unimodal_paths = ['/home/asenella/scratch/data/translated_mmnist/train/m'+str(i) for i in range(5)]

data = MMNISTDataset(unimodal_paths)

print(data[0])