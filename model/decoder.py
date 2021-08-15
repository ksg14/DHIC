from pathlib import Path
from typing import Any, Tuple
from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding, Linear
from torch import Tensor
import torch.nn.functional as F

class Decoder(Module):
	def __init__(self, n_vocab: int, emb_dim: int, padding_idx: int, dropout_p: float, enc_dim: int,
						n_head: int, dim_ff: int, activation: str, depth: int, max_len: int):
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
	

	def forward (self, batch_sz: int, caption: Tensor, caption_mask: Tensor, target: Tensor, enc_last_hidden: Tensor) -> Tuple:
		pass
	
