from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding

from dataclasses import dataclass

from model_utils import PositionalEncoding

@dataclass
class VitDecoder(Module):
	n_vocab: int
	emb_dim: int
	padding_idx: int
	dropout_p: int
	enc_dim: int
	n_head: int
	dim_ff: int
	activation: str
	depth: int
	max_len: int

	def __post_init__(self):
		super().__init__ ()

		self.emb_layer = Embedding (self.n_vocab, self.emb_dim, self.padding_idx)
		self.pos_enc = PositionalEncoding (self.emb_dim, self.dropout_p, self.max_len)
		self.decoder_layer = TransformerDecoderLayer (self.enc_dim, self.nhead, self.dim_ff, self.dropout_p, self.activation)
		self.transformer_decoder = TransformerDecoder(self.decoder_layer, self.depth)

	def forward(self, tgt, mem, tgt_mask):
		pass
