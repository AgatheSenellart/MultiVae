
from dataset import CUB
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from PIL import Image
import numpy as np

from dataset import CUB, MultimodalBaseDataset
import torch
from multivae.models import AutoModel
from multivae.metrics import Visualization, VisualizationConfig

test_data = CUB('/home/asenella/scratch/data', split='test',max_lenght=32).text_data


model = AutoModel.load_from_folder('/home/asenella/scratch/experiments/CUB/JNF_training_2024-09-20_10-59-58/final_model')


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

# Generate sentences and transform it 
nb_coherent = 0
nb_image = 0
for color in ['white', 'yellow', 'red','blue', 'green','grey', 'brown', 'black']:
    
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
    print(tokenized)
    print(idx)
    sent = torch.nn.functional.one_hot(torch.Tensor(idx).long(), test_data.vocab_size).float().unsqueeze(0)
    
    testing_set = MultimodalBaseDataset(data=dict(text = sent))
    
    # Visualize samples
    vis_config = VisualizationConfig(n_samples=10,n_data_cond=1)
    vis_module = Visualization(model,test_dataset=testing_set,output=f'./test/{color}', eval_config=vis_config)
    
    vis_module.conditional_samples_subset(['text'],gen_mod='image')
    
    
    # Compute coherence metric
    model.eval()
    output = model.predict(inputs = MultimodalBaseDataset(data=dict(text = sent)),cond_mod='text',gen_mod='image', N=10) # 10 images ce n'est pas très significatif
    
    print(output.image.shape) # 10,1, 3, 64, 64
    
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
        
        



        
        
        
    
    
    