from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from torch import nn
from torch.tensor import Tensor

from transformers import ViTFeatureExtractor, ViTModel

@dataclass
class VitEncoder(nn.Module):
	fe_path: Path
	vit_path: Path
	out_attentions: bool=False
	
	def __post_init__(self):
		super().__init__()
		self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.fe_path)
		self.vit_model = ViTModel.from_pretrained(self.vit_path)
	
	def forward (self, images_list: List[Tensor]) -> Tuple:
		image_inputs = self.feature_extractor(images=images_list, return_tensors="pt")
		enc_outputs = self.vit_model(**image_inputs, output_attentions=self.out_attentions)
		enc_last_hidden_state = enc_outputs.last_hidden_state

		if self.out_attentions:
			attentions = enc_outputs.attentions
		else:
			attentions = None

		return enc_last_hidden_state, attentions


