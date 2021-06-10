from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding, Linear
from torch.tensor import Tensor
import torch.nn.functional as F

from transformers import ElectraModel

from dataclasses import dataclass

@dataclass
class ElectraDecoder(Module):
	electra_path: str
	enc_hidden_dim: int

	def __post_init__(self):
		super().__init__ ()
		self.electra_model = ElectraModel.from_pretrained(self.electra_path, is_decoder = True)
		self.fc = Linear (self.enc_hidden_dim, self.electra_model.config.hidden_size)

	def forward (self, caption: Tensor, caption_mask: Tensor, enc_last_hidden: Tensor) -> Tensor:
		print (f'caption - {caption.shape}')
		print (f'enc hidden - {enc_last_hidden}')

		fc_out = F.gelu (self.fc (enc_last_hidden))

		print (f'fc out - {fc_out.shape}')

		return None

# @dataclass
# class VitDecoder(Module):
# 	n_vocab: int
# 	emb_dim: int
# 	padding_idx: int
# 	dropout_p: int
# 	enc_dim: int
# 	n_head: int
# 	dim_ff: int
# 	activation: str
# 	depth: int
# 	max_len: int

# 	def __post_init__(self):
# 		super().__init__ ()

# 		self.emb_layer = Embedding (self.n_vocab, self.emb_dim, self.padding_idx)
# 		self.pos_enc = PositionalEncoding (self.emb_dim, self.dropout_p, self.max_len)
# 		self.decoder_layer = TransformerDecoderLayer (self.enc_dim, self.nhead, self.dim_ff, self.dropout_p, self.activation)
# 		self.transformer_decoder = TransformerDecoder(self.decoder_layer, self.depth)

# 	def forward(self, tgt, mem, tgt_mask):
# 		pass
