import torch
import math
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Taken from torch/examples"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CubTextEncoder(BaseEncoder):
    def __init__(self, args, ntokens: int, embed_size: int = 512, nhead: int = 8, ff_size: int = 1024, n_layers : int = 8, dropout: float = 0.5):
        """
        Parameters:
            ntokens (int): Vocabulary size.
            embed_size (int): Size of the token embedding vectors. Default: 512
            nhead (int): Number of head in the MultiHeadAttention module. Default: 8
            ff_size (int): Number of units in the feedforward layers. Default: 1024
            n_layers (int): Number of Encoders layers in the TransformerEncoder. Default: 8
            dropout (float): Dropout rate. Default: 0.5
        """
        
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim # input_dim = [max_sentence_length, vocab_size]
        self.latent_dim = args.latent_dim

        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(ntokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead, ff_size, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.mu = nn.Linear(np.prod(self.input_dim), self.latent_dim)
        self.log_covariance = nn.Linear(np.prod(self.input_dim), self.latent_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.token_embedding.weight, -initrange, initrange)

    def forward(self, src, padding_mask):
    
        src = self.token_embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = ModelOutput(
            embedding=self.mu(transformer_output.reshape(src.shape[0], -1)),
            log_covariance=self.log_covariance(transformer_output.reshape(src.shape[0], -1)))
        return output