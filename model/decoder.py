from pathlib import Path
from typing import Tuple
from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding, Linear
from torch import Tensor
import torch.nn.functional as F

from transformers import ElectraModel, ElectraForMaskedLM, BertLMHeadModel

from dataclasses import dataclass

class Decoder(Module):
	def __init__(self, decoder_path: Path, out_attentions: bool=False):
		super().__init__()
		self.out_attentions = out_attentions
		
		self.model = BertLMHeadModel.from_pretrained(decoder_path, is_decoder = True, add_cross_attention=True)
		# self.fc = Linear (self.enc_hidden_dim, self.electra_model.config.hidden_size)

	def forward (self, batch_sz: int, caption: Tensor, caption_mask: Tensor, target: Tensor, enc_last_hidden: Tensor) -> Tuple:
		# fc_out = F.gelu (self.fc (enc_last_hidden))

		outputs = self.model (input_ids=caption.view (batch_sz, -1), attention_mask=caption_mask.view (batch_sz, -1), encoder_hidden_states=enc_last_hidden, labels=target.view (batch_sz, -1), output_attentions=self.out_attentions)

		if self.out_attentions:
			attentions = outputs.attentions
		else:
			attentions = None
		
		return outputs.loss, outputs.logits, attentions
	
	def save_model (self, save_path: Path) -> None:
		self.model.save_pretrained (save_path)

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
