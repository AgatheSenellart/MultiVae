import io
import json
import logging
import os
import pickle
import tempfile
from collections import Counter, OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

from multivae.data.datasets.base import DatasetOutput, MultimodalBaseDataset

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSentences(Dataset):  # pragma: no cover
    """

    Dataset for the CUB captions only.

    Args:
        - root_data_dir (str): The path where to find the data.
        - split (str): The split to use. Either 'trainval' or 'test'.
        - output_type (str): The output type of the text. Either 'one_hot','tokens', 'text'. If text, the output is the list of words.
        - transform (torchvision.transforms): The transformations to apply to the text data. Default: None.


    """

    def __init__(
        self, root_data_dir, split, output_type="one_hot", transform=None, **kwargs
    ):
        super().__init__()

        self.data_dir = os.path.join(root_data_dir, "cub")
        self.split = split
        self.max_sequence_length = kwargs.get("max_sequence_length", 32)
        self.min_occ = kwargs.get("min_occ", 3)
        self.transform = transform
        self.output_type = output_type

        self.gen_dir = os.path.join(
            self.data_dir, "oc:{}_msl:{}".format(self.min_occ, self.max_sequence_length)
        )

        if split == "train":
            self.raw_data_path = os.path.join(self.data_dir, "text_trainvalclasses.txt")
        elif split == "test":
            self.raw_data_path = os.path.join(self.data_dir, "text_testclasses.txt")
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = "cub.{}.s{}".format(split, self.max_sequence_length)
        self.vocab_file = "cub.vocab"

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print(
                "Data file not found for {} split at {}. Creating new... (this may take a while)".format(
                    split.upper(), os.path.join(self.gen_dir, self.data_file)
                )
            )
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = torch.LongTensor(self.data[str(idx)]["idx"])
        length = self.data[str(idx)]["length"]
        padding_mask = torch.FloatTensor(
            [1.0] * length + [0.0] * (self.max_sequence_length - length)
        )

        if self.output_type == "one_hot":
            sent = nn.functional.one_hot(
                torch.Tensor(sent).long(), self.vocab_size
            ).float()
            if self.transform is not None:
                sent = self.transform(sent)
            return dict(one_hot=sent, padding_mask=padding_mask)

        elif self.output_type == "tokens":
            if self.transform is not None:
                sent = self.transform(sent)
            return dict(tokens=sent, padding_mask=padding_mask)

        else:
            raise AttributeError(
                "The output type should be either 'one_hot' or 'tokens' but is neither."
            )

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    @property
    def unk_idx(self):
        return self.w2i["<unk>"]

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        try:
            with open(os.path.join(self.gen_dir, self.data_file), "rb") as file:
                self.data = json.load(file)
        except TypeError:
            with open(os.path.join(self.gen_dir, self.data_file), "r") as file:
                self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab["w2i"], vocab["i2w"]

    def _create_data(self):
        if self.split == "train" and not os.path.exists(
            os.path.join(self.gen_dir, self.vocab_file)
        ):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, "r") as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[: self.max_sequence_length - 1]
            tok = tok + ["<eos>"]
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(["<pad>"] * (self.max_sequence_length - length))
                pad_count += 1
            else:
                length = self.max_sequence_length
            idx = [self.w2i.get(w, self.w2i["<exc>"]) for w in tok]

            id = len(data)
            data[id]["tok"] = tok
            data[id]["idx"] = idx
            data[id]["length"] = length

        print(
            "{} out of {} sentences are truncated with max sentence length {}.".format(
                len(sentences) - pad_count, len(sentences), self.max_sequence_length
            )
        )
        with io.open(os.path.join(self.gen_dir, self.data_file), "wb") as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode("utf8", "replace"))

        self._load_data(vocab=False)

    def _create_vocab(self):
        import nltk

        nltk.download("punkt_tab")
        nltk.download("punkt")

        assert (
            self.split == "train"
        ), "Vocablurary can only be created for training file."

        with open(self.raw_data_path, "r") as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ["<exc>", "<pad>", "<eos>"]
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

        print(
            "Vocablurary of {} keys created, {} words are excluded (occurrence <= {}).".format(
                len(w2i), len(unq_words), self.min_occ
            )
        )

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), "wb") as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode("utf8", "replace"))

        with open(os.path.join(self.gen_dir, "cub.unique"), "wb") as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, "cub.all"), "wb") as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()

    def one_hot_to_string(self, data):
        ret_list = [self._to_string(i) for i in data]
        return ret_list

    def _to_string(self, matrix):
        words = []

        for i in range(matrix.shape[0]):
            idx = np.argmax(matrix[i, :])
            words.append(self.i2w[str(idx)])

        ret_str = " ".join(words)
        return ret_str


class CUB(MultimodalBaseDataset):  # pragma: no cover
    """

    A paired text image CUB dataset.

    Args:
        path (str) : The path where the data is saved.
        split (str) : Either 'train', 'eval' or 'test'. Default: 'train'.
        max_words_in_caption (int): The number of words in the captions. Default: 32.
        im_size (Tuple[int]): The desired size of the images. Default: (64, 64)
        img_transform (Transforms): The transformations to be applied to the images. If
            None, nothing is done. Default: None.
        output_type (Literal['one_hot','tokens']) : Default to 'one_hot'.
        download (bool): Whether to download the data if it is not found in the path. Default: False.
    """

    def __init__(
        self,
        path: str,
        split="train",
        max_words_in_caption=32,
        im_size=(64, 64),
        img_transform=None,
        output_type="one_hot",
        download=False,
    ):
        self.split = split
        self.check_or_download_data(path, download)

        self.max_words_in_caption = max_words_in_caption

        transform_text = lambda data: torch.Tensor(data)
        transform_img = [transforms.Resize([*im_size]), transforms.ToTensor()]
        if img_transform is not None:
            transform_img.append(img_transform)
        transform_img = transforms.Compose(transform_img)

        if split == "eval":
            self.text_data = CUBSentences(
                path,
                "train",
                output_type=output_type,
                transform=transform_text,
                max_sequence_length=max_words_in_caption,
            )
            self.image_data = datasets.ImageFolder(
                os.path.join(path, "cub", "train"), transform=transform_img
            )

        else:
            self.text_data = CUBSentences(
                path,
                split,
                output_type=output_type,
                transform=transform_text,
                max_sequence_length=max_words_in_caption,
            )
            self.image_data = datasets.ImageFolder(
                os.path.join(path, "cub", split), transform=transform_img
            )

        # Split the training data into train and validation
        if self.split == "train" or "eval":
            self.train_idx, self.val_idx = train_test_split(
                np.arange(len(self.text_data)),
                test_size=0.1,
                random_state=0,
                shuffle=True,
            )

        self.vocab_size = self.text_data.vocab_size

    def check_or_download_data(self, data_path, download):
        if not os.path.exists(os.path.join(data_path, "cub")):
            if download:
                os.makedirs(os.path.join(data_path, "cub"))
                tempdir = tempfile.mkdtemp()
                logger.info(f"Downloading the CUB dataset into {data_path}")
                download_and_extract_archive(
                    url="http://www.robots.ox.ac.uk/~yshi/mmdgm/datasets/cub.zip",
                    download_root=tempdir,
                    extract_root=data_path,
                )
            else:
                raise AttributeError(
                    "The CUB dataset is not available at the"
                    " given datapath and download is set to False."
                    "Set download to True or place the dataset"
                    " in the data_path folder."
                )

    def __getitem__(self, index):
        if self.split == "train":
            index = self.train_idx[index]
        elif self.split == "eval":
            index = self.val_idx[index]

        image = self.image_data[index // 10][0]
        text = self.text_data[index]

        return DatasetOutput(data=dict(image=image, text=text))

    def __len__(self):
        if self.split == "train":
            return len(self.train_idx)

        if self.split == "eval":
            return len(self.val_idx)

        else:
            return len(self.text_data)

    def plot_text(self, input_tensor, fig_size=(2, 1.5)):
        # input_tensor is of shape (max_sequence_lenght,vocab_size)
        device = input_tensor.device
        array = input_tensor.detach().cpu().numpy()
        sentence = self.text_data._to_string(array)

        fig = plt.figure(figsize=fig_size)
        plt.text(
            x=0.5,
            y=0.5,
            s="{}".format(
                " ".join(
                    i + "\n" if (n + 1) % 3 == 0 else i
                    for n, i in enumerate(
                        [word for word in sentence.split() if word != "<eos>"]
                    )
                )
            ),
            fontsize=7,
            verticalalignment="center_baseline",
            horizontalalignment="center",
        )
        plt.axis("off")
        fig.tight_layout()
        # Draw the canvas and retrieve the image as a NumPy array
        fig.canvas.draw()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        image = PIL.Image.open(img_buf)

        image = np.array(image).transpose(2, 0, 1) / 255
        plt.close(fig=fig)
        return torch.from_numpy(image).float().to(device)

    def transform_for_plotting(self, input, modality):
        """Transform the data for plotting purposes

        args :

            input (dict or tensor) : the input has the same type as returned by the getitem method for each modality type.
            modality (str) : the name of the modality"""

        if modality == "text":
            list_transformed = []
            if isinstance(input, torch.Tensor):
                for x in input:
                    list_transformed.append(self.plot_text(x))
                return torch.stack(list_transformed)
            elif isinstance(input, dict):
                # The input is a dict with either a field 'one_hot' or a field 'tokens'
                if "one_hot" in input:
                    tensor = input["one_hot"]
                    for x in tensor:
                        list_transformed.append(self.plot_text(x))
                elif "tokens" in input:
                    tensor = input["tokens"]
                    for x in tensor:
                        x = nn.functional.one_hot(
                            torch.Tensor(x).long(), self.text_data.vocab_size
                        ).float()
                        list_transformed.append(self.plot_text(x))
                else:
                    raise AttributeError(
                        'The text input should be a dictionary with either "one_hot" or "tokens" as a key but it has neither.'
                    )
                return torch.stack(list_transformed)
            else:
                raise AttributeError(
                    "The input should be either a tensor or a dictionary but is neither"
                )

        if modality == "image":
            return input
