from dataclasses import dataclass
from pathlib import Path
from typing import List
from torch import nn
from torch.tensor import Tensor

from transformers import ViTFeatureExtractor, ViTModel

@dataclass
class VitEncoder(nn.Sequential):
	fe_path: Path
	vit_path: Path
	
	def __post_init__(self):
		super().__init__()
		self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.fe_path)
		self.vit_model = ViTModel.from_pretrained(self.vit_path)
	
	def forward (self, images_list: List[Tensor]) -> Tensor:
		image_inputs = self.feature_extractor(images=images_list, return_tensors="pt")
		enc_outputs = self.vit_model(**image_inputs, output_attentions=False, output_hidden_states=False)
		enc_last_hidden_state = enc_outputs.last_hidden_state
		return enc_last_hidden_state


