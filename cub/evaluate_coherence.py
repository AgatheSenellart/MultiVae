
from dataset import CUB
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import PIL

from PIL import Image
import numpy as np

from dataset import CUB, MultimodalBaseDataset, DatasetOutput
import torch
from multivae.models import AutoModel
from multivae.metrics import Visualization, VisualizationConfig
import os
import wandb
from multivae.trainers.base.callbacks import load_wandb_path_from_folder
from torch.utils.data import DataLoader

def in_range(hsv, lower, upper):
    mask_h = np.logical_and(hsv[:,:,0] >= lower[0], hsv[:,:,0] <= upper[0])
    mask_s = np.logical_and(hsv[:,:,1] >= lower[1] , hsv[:,:,1] <= upper[1])
    mask_v = np.logical_and(hsv[:,:,2] >= lower[2] , hsv[:,:,2] <= upper[2])
    
    return mask_h & mask_s & mask_v

def count_pixels_per_color(hsv_image):
    
    color_ranges = dict(
        white = lambda h : in_range(h,[0,0,120],[180,18,255]),
        yellow = lambda h : in_range(h,[25,50,70],[35,255,255]),
        blue = lambda h : in_range(h,[90,50,70],[158,255,255]),
        green = lambda h: in_range(h,[36, 50, 70],[89, 255, 255]),
        gray = lambda h: in_range(h,[0, 0, 50],[180, 18, 120]),
        brown = lambda h: in_range(h,[16, 50, 70],[24, 255, 255]),
        black = lambda h: in_range(h,[0, 0, 0],[180, 255, 50]),
        red = lambda h: np.logical_or(in_range(h,[0, 50, 70],[15, 255, 255]), in_range(h,[159, 50, 70],[180, 255, 255]))
    )
    
    color_masks = dict()
    color_count = dict()
    for color in color_ranges:
        color_masks[color] = color_ranges[color](hsv_image)
        color_count[color] = color_masks[color].sum()
        
    # get the two most present colors :
    
    values = np.sort(list(color_count.values()))[-2:]
    
    most_present = [color for color in color_count if color_count[color] in values]
        
    return color_masks, color_count, most_present


class simple_text_dataset(CUB):
    
    def __init__(self, root_data_dir, split='train', max_lenght=32, one_hot=True,color='blue'):
        super().__init__(root_data_dir, split, max_lenght, one_hot)
        self.color = color
        
    def __getitem__(self, index):
        
        color = self.color
        sentence = f'This bird is completely {color}'
        
        tokenized = word_tokenize(sentence)
        
        # Then we need to transform to the same format as in the CUB dataset
        tok = tokenized + ['<eos>']
        length = len(tok)
        pad_count = 0
        if test_data.max_sequence_length > length:
            tok.extend(['<pad>'] * (test_data.max_sequence_length - length))
            pad_count += 1
        idx = [test_data.w2i.get(w, test_data.w2i['<exc>']) for w in tok]

        sent = torch.nn.functional.one_hot(torch.Tensor(idx).long(), test_data.vocab_size).float()
        sent = dict(one_hot = sent)
        
        return DatasetOutput(data = dict(text = sent))
        
    def __len__(self):
        return 10
    
    

def evaluate_coherence(model, wandb_path, test_data):
    
    entity, project, run_id = tuple(wandb_path.split("/"))
    wandb_run = wandb.init(
                project="mmvae_plus_CUB", id=run_id, resume="must", reinit=True
        )
    # Generate sentences and transform it 
    nb_coherent = 0
    nb_image = 0
    for color in ['white', 'yellow', 'red','blue', 'green','grey', 'brown', 'black']:
        
        testing_set = simple_text_dataset('/home/asenella/scratch/data','test',32,True,color)
        
        # Visualize samples
        vis_config = VisualizationConfig(n_samples=10,n_data_cond=1)
        vis_module = Visualization(model,test_dataset=testing_set,output=f'./test/{color}', eval_config=vis_config)
        
        recon_image = vis_module.conditional_samples_subset(['text'],gen_mod='image')
        
        wandb_run.log(
            {f"conditional_from_text": wandb.Image(recon_image)}
        )

        
        # Compute coherence metric
        model.eval()
        inputs = next(iter(DataLoader(testing_set,batch_size=1)))
        output = model.predict(inputs =inputs ,cond_mod='text',gen_mod='image', N=10) # 10 images ce n'est pas très significatif
        
        # print(output.image.shape) # 10,1, 3, 64, 64
        
        # for each image, we need to evaluate the main colors
        
        for im in output.image:
            nb_image +=1
            
            # Transform to PIL 
            ndarr = (
                im[0].mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            recon_image = Image.fromarray(ndarr).convert('HSV')
            
            
            hsv = np.array(recon_image)[:, :, :].copy()
            
            color_masks, color_count, most_present = count_pixels_per_color(hsv)
            if color in most_present:
                nb_coherent +=1


    print('coherence :', nb_coherent/nb_image)
    wandb.log({'coherence': nb_coherent/nb_image})
    wandb_run.finish()



if __name__ == '__main__':
    
    test_data = CUB('/home/asenella/scratch/data', split='test',max_lenght=32).text_data
    for path in os.listdir('/home/asenella/experiments/CUB_new'):
        print( path)
        path_model = os.path.join('/home/asenella/experiments/CUB_new',path,'final_model')
        if os.path.exists(path_model):
            print('starting evaluation')  
            model = AutoModel.load_from_folder(path_model)
            wandb_path = load_wandb_path_from_folder(path_model)
            
            evaluate_coherence(model, wandb_path, test_data)
        
        

        



        
        
        
    
    
    