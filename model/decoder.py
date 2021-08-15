from pathlib import Path
from typing import Any, Tuple
from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding, Linear, Dropout
from torch import Tensor
import torch.nn.functional as F
import math
import torch

class Decoder(Module):
	def __init__(self, n_vocab: int, emb_dim: int, padding_idx: int, dropout_p: float, enc_dim: int,
						n_head: int, n_layers: int, dim_ff: int, activation: str, depth: int, max_len: int, device: str):
		super().__init__()
		self.n_vocab = n_vocab
		self.emb_dim = emb_dim 
		self.padding_idx = padding_idx
		self.dropout_p = dropout_p 
		self.enc_dim = enc_dim 
		self.n_head = n_head 
		self.dim_ff = dim_ff 
		self.activation = activation
		self.depth = depth 
		self.max_len = max_len

		# Token embedding
		self.word_embedding = Embedding(n_vocab, emb_dim, padding_idx=0)

		# Positional Embedding
		position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_emb', pe)
	
		# Decoder
		self.decoder_layer = TransformerDecoderLayer(d_model=emb_dim, nhead=n_head)
		self.decoder = TransformerDecoder(self.decoder_layer, num_layers=n_layers)

	def forward (self, batch_sz: int, caption: Tensor, caption_mask: Tensor, target: Tensor, 
					enc_last_hidden: Tensor) -> Tuple:
		pass
	
