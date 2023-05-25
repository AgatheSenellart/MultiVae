
# Get all the wandb info and push to hfhub

model_paths = '/Users/agathe/Desktop/models/compare_on_mmnist/config2/JMVAE'


for seed in [0,1,2,3]:
    for missing_ratio in [0,0.2,0.5]:
        
        path = model_paths + f'/seed_{seed}/missing_ratio_{missing_ratio}/' 
        
        