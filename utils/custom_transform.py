from typing import Callable, List
import torch

def prepare_sequence (seq : str, to_ix : dict, tokenizer : Callable) -> torch.Tensor:
	idxs = [to_ix[w] for w in tokenizer (seq)]
	return torch.tensor(idxs, dtype=torch.long)

class ToSequence(object):
	def __init__(self, tokenizer : Callable) -> None:
		self.tokenizer = tokenizer

	def __call__(self, seq : str, wtoi : dict) -> torch.Tensor:
		return prepare_sequence(seq, wtoi, self.tokenizer)

