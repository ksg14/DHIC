from torch.nn import Module, TransformerDecoderLayer, TransformerDecoder, Embedding

from model_utils import PositionalEncoding

class VitDecoder(Module):
	def __init__(self, depth: int, enc_dim: int, nhead: int, dim_ff: int, dropout_p: float, activation: str, n_vocab: int, emb_dim: int, padding_idx: int, max_len: int):
		super().__init__ ()
		self.enc_dim = enc_dim
		self.depth = depth
		self.nhead = nhead
		self.dim_ff = dim_ff
		self.dropout_p = dropout_p
		self.activation = activation
		self.n_vocab = n_vocab
		self.emb_dim = emb_dim
		self.padding_idx = padding_idx

		self.emb_layer = Embedding (self.n_vocab, self.emb_dim, self.padding_idx)
		self.pos_enc = PositionalEncoding (emb_dim, dropout_p, max_len)
		self.decoder_layer = TransformerDecoderLayer (self.enc_dim, self.nhead, self.dim_ff, self.dropout_p, self.activation)
		self.transformer_decoder = TransformerDecoder(self.decoder_layer, self.depth)

	def forward(self, tgt, mem, tgt_mask):
		pass
