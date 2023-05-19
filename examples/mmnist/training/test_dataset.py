from config2 import *

import numpy as np

train_data = MMNISTDataset(data_path="~/scratch/data/MMNIST", 
                           split="train", 
                           missing_ratio=0.5,
                           keep_incomplete=False)

labels = []
for i in range(len(train_data)):
    print(train_data[i].labels)
    labels.append(train_data[i].labels.item())
    
for k in range(10):
    count = np.sum(np.array(labels) == k)/len(labels)
    print('k :  ',k, count)