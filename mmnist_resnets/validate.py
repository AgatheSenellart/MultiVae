

from global_config import *
from multivae.models import AutoModel
from multivae.data.datasets import MMNISTDataset
import time
from multivae.trainers.base.callbacks import load_wandb_path_from_folder
import os

train_data = MMNISTDataset(
    data_path="~/scratch/data",
    split="train"
)

test_data = MMNISTDataset(data_path="~/scratch/data", split="test")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 0

def eval_model(model, output_dir, train_data,test_data, wandb_path, seed):
    """
    In this function, define all the evaluation metrics
    you want to use
    """
    global fid_path
    # config = CoherenceEvaluatorConfig(batch_size=128, wandb_path=wandb_path)
    # mod = CoherenceEvaluator(
    #     model=model,
    #     test_dataset=test_data,
    #     classifiers=load_mmnist_classifiers(device=model.device),
    #     output=output_dir,
    #     eval_config=config,
    # )
    # mod.eval()
    # mod.finish()
     # FID evaluator
    config = FIDEvaluatorConfig(batch_size=128, wandb_path=wandb_path, inception_weights_path=fid_path)

    fid = FIDEvaluator(
        model, test_data, output=output_dir, eval_config=config
    )
    fid.compute_all_conditional_fids(gen_mod="m0")
    fid.finish()
    



main_path = '/home/asenella/scratch/experiments_/experiments/mmnist_resnets/JNF/seed_0/'
# l = os.listdir(main_path)

l = ['JNF_training_2024-10-23_14-09-32/'
    #  'MVTCAE_training_2024-10-23_11-29-19'
     ]

for path in l :
    dir_ = os.path.join(main_path,path,'final_model')
    
    if os.path.exists(dir_):
            
        # dir = '/home/asenella/experiments/mmnist_resnets/MoPoE/seed_0/MoPoE_training_2024-09-20_17-33-05/final_model'
        wandb_path = load_wandb_path_from_folder(dir_)

        model = AutoModel.load_from_folder(dir_)
        model = model.eval().to(device)
        model.device = device

        
        t1 = time.time()
        eval_model(model, dir_,train_data=train_data, test_data=test_data,wandb_path=wandb_path,seed=seed)
        t2 = time.time()
        print(t2-t1)