from pathlib import Path
from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.types import Device

from transformers import ViTFeatureExtractor, ViTModel

class VitEncoder(Module):
	def __init__(self, fe_path: Path, vit_path: Path, device: Device, out_attentions: bool=False, do_resize: bool=False, do_normalize: bool=False):
		super().__init__()
		self.out_attentions = out_attentions
		self.do_resize = do_resize
		self.do_normalize = do_normalize
		self.device = device

		self.feature_extractor = ViTFeatureExtractor.from_pretrained(fe_path, do_resize=self.do_resize, 
																		do_normalize=self.do_normalize)
		
		# print (f'fe img mean - {self.feature_extractor.image_mean}')
		# print (f'fe img std - {self.feature_extractor.image_std}')

		self.vit_model = ViTModel.from_pretrained(vit_path)

	def forward (self, images_list: List[Tensor]) -> Tuple:
		image_inputs = self.feature_extractor(images=images_list, return_tensors="pt").to (self.device)
		enc_outputs = self.vit_model(**image_inputs, output_attentions=self.out_attentions)
		# enc_last_hidden_state = enc_outputs.last_hidden_state

		# if self.out_attentions:
		# 	attentions = enc_outputs.attentions
		# else:
		# 	attentions = None

		# return enc_last_hidden_state, attentions
		return enc_outputs
	
	def save_model (self, fe_save_path: Path, vit_save_path: Path) -> None:
		self.feature_extractor.save_pretrained (fe_save_path)
		self.vit_model.save_pretrained (vit_save_path)


