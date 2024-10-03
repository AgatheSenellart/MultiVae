# from multivae.data.datasets import MultimodalBaseDataset
import io
import json
import os
import pickle
from collections import Counter, OrderedDict
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PIL
from pythae.data.datasets import DatasetOutput

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSentences(Dataset):

    def __init__(self, root_data_dir, split, one_hot=False, transpose=False, transform=None, **kwargs):
        """split: 'trainval' or 'test' """

        super().__init__()
        self.data_dir = os.path.join(root_data_dir, 'cub')
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 32)
        self.min_occ = kwargs.get('min_occ', 3)
        self.transform = transform
        self.one_hot = one_hot
        self.transpose = transpose
        # os.makedirs(os.path.join(root_data_dir, "lang_emb"), exist_ok=True)

        self.gen_dir = os.path.join(self.data_dir, "oc:{}_msl:{}".
                                    format(self.min_occ, self.max_sequence_length))

        if split == 'train':
            self.raw_data_path = os.path.join(self.data_dir, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.raw_data_path = os.path.join(self.data_dir, 'text_testclasses.txt')
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'cub.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = torch.LongTensor(self.data[str(idx)]['idx'])
        length = self.data[str(idx)]['length']
        
        padding_mask=torch.FloatTensor([1.0]*length + [0.]*(self.max_sequence_length-length))

        if self.one_hot:
            sent = nn.functional.one_hot(torch.Tensor(sent).long(), self.vocab_size).float()

        if self.transpose:
            sent = sent.transpose(-2, -1)
        if self.transform is not None:
            sent = self.transform(sent)
        
        if self.one_hot :
            
            return dict(one_hot = sent, padding_mask=padding_mask)
        else :
            return dict(tokens = sent, padding_mask=padding_mask)


    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        try:
            with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
                self.data = json.load(file)
        except TypeError:
            with open(os.path.join(self.gen_dir, self.data_file), 'r') as file:
                self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            else :
                length = self.max_sequence_length
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        import nltk
        nltk.download('punkt')

        assert self.split == 'train', "Vocablurary can only be created for training file."

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()

    def one_hot_to_string(self, data):
        ret_list = [self._to_string(i) for i in data]
        return ret_list

    def _to_string(self, matrix):
        words = []

        if self.transpose:
            matrix = matrix.T

        for i in range(matrix.shape[0]):
            idx = np.argmax(matrix[i, :])
            words.append(self.i2w[str(idx)])

        ret_str = " ".join(words)
        return ret_str

# This is required because there are 10 captions per image.
# Allows easier reuse of the same image for the corresponding set of captions.
def resampler(dataset, idx):
    return idx // 10


from multivae.data.datasets.base import MultimodalBaseDataset, DatasetOutput
from torchvision import transforms, datasets


class CUB(MultimodalBaseDataset):
    
    def __init__(self, root_data_dir, split='train',max_lenght = 32, one_hot=True):
        
        
        self.one_hot = one_hot
        self.split = split
        transform_text = lambda data: torch.Tensor(data)
        tx = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

        if split == 'eval':
            self.text_data = CUBSentences(root_data_dir,'train',one_hot=one_hot,transpose=False,transform=transform_text, max_sequence_length=max_lenght)
            self.image_data = datasets.ImageFolder(os.path.join(root_data_dir, 'cub','train'), transform=tx)

        else :
            self.text_data = CUBSentences(root_data_dir,split,one_hot=True,transpose=False,transform=transform_text, max_sequence_length=max_lenght)
            self.image_data = datasets.ImageFolder(os.path.join(root_data_dir, 'cub',split), transform=tx)

        
        if self.split == 'train' or 'eval':
            self.train_idx, self.val_idx = train_test_split(np.arange(len(self.text_data)),test_size=0.1, random_state = 0,shuffle=True)
             
            
            
        
    def __getitem__(self, index):
        
        if self.split == 'train':
            index = self.train_idx[index]
        elif self.split == 'eval':
            index = self.val_idx[index]
        
        image = self.image_data[index // 10][0]
        text = self.text_data[index]
        
        return DatasetOutput(data = dict(
            image = image,
            text = text)
        )
    
    def __len__(self):
        
        if self.split == 'train':
            return len(self.train_idx)

        if self.split == 'eval':
            return len(self.val_idx)
        
        else :
            return len(self.text_data)


    def plot_text(self,input_tensor, fig_size=(2,1.5)):
        # input_tensor is of shape (max_sequence_lenght,vocab_size)
        device = input_tensor.device
        array = input_tensor.detach().cpu().numpy()
        sentence = self.text_data._to_string(array)
        
        fig = plt.figure(figsize=fig_size)
        plt.text(
            x=0.5,
            y=0.5,
            s='{}'.format(
                ' '.join(i + '\n' if (n + 1) % 3 == 0
                            else i for n, i in enumerate([word for word in sentence.split() if word != '<eos>']))),
            fontsize=7,
            verticalalignment='center_baseline',
            horizontalalignment='center'
        )
        plt.axis('off')
        fig.tight_layout()
        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        image = PIL.Image.frombytes('RGB', 
                    fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        
        image= np.array(image).transpose(2,0,1)/255
        plt.close(fig=fig)
        return torch.from_numpy(image).float().to(device)
    
    
    def transform_for_plotting(self, input, modality):
        ''' Transform the data for plotting purposes
        
        args :
        
            input (dict or tensor) : the input has the same type as returned by the getitem method for each modality type.
            modality (str) : the name of the modality'''
            
        if modality == 'text':
            if not isinstance(input,dict):
                raise AttributeError('The input for modality = "text" should be a dictionary but it is of type :', type(input))
            list_transformed = []
            # The input is a dict with either a field 'one_hot' or a field 'tokens'
            if 'one_hot' in input:
                tensor = input['one_hot']
                for x in tensor: 
                    list_transformed.append(self.plot_text(x))
            elif 'tokens' in input:
                tensor = input['tokens']
                for x in tensor:
                    x = nn.functional.one_hot(torch.Tensor(x).long(), self.text_data.vocab_size).float()
                    list_transformed.append(self.plot_text(x))
            else:
                raise AttributeError('The text input should be a dictionary with either "one_hot" or "tokens" as a key but it has neither.')
            return torch.stack(list_transformed)
                
        if modality == 'image' :
            return input
        
        
        
    