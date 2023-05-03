import os
from pathlib import Path
from typing import Union
import numpy as np

import torch
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms

from .base import MultimodalBaseDataset
from .utils import ResampleDataset
from collections import defaultdict
from pythae.data.datasets import DatasetOutput

import pickle
import pandas as pd
from PIL import Image
from nltk.tokenize import RegexpTokenizer


class CUB(MultimodalBaseDataset):
    """

    A paired text img CUB dataset.

    Args:
        path (str) : The path where the data is saved.
        split (str) : Either 'train' or 'test'.
        download (bool) : Whether to download the data or not. Default to True.
        data_multiplication (int) : 

        **kwargs:

            transform_mnist (Transform) : a transformation to apply to MNIST. If none specified, a simple ToTensor() is applied.
            transform_svhn (Transform) : a transformation to apply to SVHN. If none specified, a simple ToTensor() is applied.

    """

    def __init__(
        self,
        data_path: Union[str, Path] = "../data/",
        split: str = "train",
        captions_per_image: int = 10,
        max_words_in_caption=18,
        im_size: int = (64, 64),
        img_transform=None,
        download=False,
        data_multiplication=5,
        **kwargs,
    ):  
        if split not in ["train", "test"]:
            raise AttributeError("Possible values for split are 'train' or 'test'")
        
        if captions_per_image > 10:
            raise AttributeError("Maximum number of captions per image is 10.")
        
        self.img_transform = img_transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.captions_per_img = captions_per_image
        self.data_path = data_path
        self.split = split
        self.imsize = im_size
        self.max_words_in_captions = max_words_in_caption
        self.tokenizer = RegexpTokenizer(r'\w+')

        self.train_test_split = self._load_train_test_split()
        train_filenames, test_filenames = self._load_filenames()
        train_captions = self._load_captions(train_filenames.name)
        test_captions = self._load_captions(test_filenames.name)
        self.bbox = self._load_bbox() 

        train_captions_new, test_captions_new, idxtoword, wordtoidx, vocab_size = self.build_vocab(
            train_captions,
            test_captions
        )
        
        self.train_filenames = train_filenames
        self.test_filenames = test_filenames
        self.train_captions = train_captions_new
        self.test_captions = test_captions_new
        self.idxtoword = idxtoword
        self.wordtoidx = wordtoidx
        self.vocab_size = vocab_size
        self.labels = None

    def _load_train_test_split(self):
        train_test_split = pd.read_csv(
            os.path.join(self.data_path, "train_test_split.txt"),
            delim_whitespace=True,
            header=None,
            names=["id", "split"]
        )

        return train_test_split
    
    def _load_filenames(self):
        filenames = pd.read_csv(
            os.path.join(self.data_path, "images.txt"),
            delim_whitespace=True,
            header=None,
            names=["id", "name"]
        )
        filenames["name"] = filenames["name"].str.replace('.jpg', '')
        
        train_filenames = filenames[self.train_test_split.split==1]
        test_filenames = filenames[self.train_test_split.split==0]

        train_filenames.reset_index(drop=True, inplace=True)
        test_filenames.reset_index(drop=True, inplace=True)
    
        return train_filenames, test_filenames


    def _load_captions(self, filenames):
        all_captions = defaultdict(list)
        for filename in filenames:
            cap_path = os.path.join(f'{self.data_path}', 'text', f'{filename}.txt')
            with open(cap_path, "r") as f:
                captions = f.read().splitlines()

                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions[filename].append(tokens_new)
                    cnt += 1
                    if cnt == self.captions_per_img:
                        break
                if cnt < self.captions_per_img:
                    print('ERROR: the captions for %s less than %d'
                          % (filename, cnt))

        return all_captions

    def _load_bbox(self):
        data_dir = self.data_path
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox


    def get_imgs(self, img_path, bbox=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if self.img_transform is not None:
            img = self.img_transform(img)
            
        re_img = transforms.Resize(self.imsize)(img)        
        ret = self.normalize(re_img)

        return ret
    

    def build_vocab(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = [cap for cap_list in train_captions.values() for cap in cap_list] + \
            [cap for cap_list in test_captions.values() for cap in cap_list]
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = defaultdict(list)
        for cap_key in train_captions.keys():
            for t in train_captions[cap_key]:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                train_captions_new[cap_key].append(rev)

        test_captions_new = defaultdict(list)
        for cap_key in test_captions.keys():
            for t in test_captions[cap_key]:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                test_captions_new[cap_key].append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def get_caption(self, sent_ix, captions):
        # a list of indices for a sentence
        sent_caption = np.asarray(captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.max_words_in_captions, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.max_words_in_captions:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.max_words_in_captions]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.max_words_in_captions
        return x, x_len

    def __getitem__(self, index):
        if self.split == "train":
            names = self.train_filenames.name[index]
            captions = self.train_captions[names]

        else:
            names = self.test_filenames.name[index]
            captions = self.test_captions[names]
            
        
        bbox = self.bbox[names]
        img_path = os.path.join(self.data_path, 'images', f'{names}.jpg')
        imgs = self.get_imgs(img_path, bbox=bbox)

        sent_ix = np.random.randint(0, self.captions_per_img)

        #new_sent_ix = index * self.captions_per_img + sent_ix
        caps, cap_len = self.get_caption(sent_ix, captions)

        X = dict(img=imgs, text=caps)
        y = self.labels[index] if self.labels is not None else None

        return DatasetOutput(data=X, labels=y)