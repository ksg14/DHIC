from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
# import cv2

import os
import numpy as np
import json

class HVGDataset (Dataset):
	def __init__ (self, captions_file: Path, word_to_index_file: Path, index_to_word_file: Path, images_path: Path, max_len : int, text_transform: Callable=None, tokenizer: Callable=None, decoder_transform: Callable=None, image_transform: Callable=None) -> None:
		with open (captions_file, 'r') as file_io:
			self.captions = json.load (file_io)
		
		with open (word_to_index_file, 'r') as file_io:
			self.word_to_index = json.load (file_io)
		
		with open (index_to_word_file, 'r') as file_io:
			self.index_to_word = json.load (file_io)
		
		self.max_len = max_len
		self.images_path = images_path
		self.text_transform = text_transform
		self.tokenizer = tokenizer
		self.decoder_transform = decoder_transform
		self.image_transform = image_transform

	def __len__ (self) -> int:
		return len (self.captions ['annotations'])
		
	def __getitem__ (self, idx: int) -> Tuple:
		image_id = self.captions ['annotations'] [idx] ['image_id']
		caption_str = self.captions ['annotations'] [idx] ['caption']
		# Image
		image_file = os.path.join (self.images_path, f'{image_id}.jpg')
		# image_file = os.path.join (self.images_path, f'16.jpg')
		image = Image.open (image_file).convert('RGB')
		if self.image_transform:
			image = self.image_transform (image)

		if self.decoder_transform:
			if self.tokenizer:
				caption_tok = self.tokenizer (caption_str)
			else:
				caption_tok = self.decoder_transform.tokenize (caption_str)

			# print (f'caption - {caption_tok [:-1]}')
			# print (f'target - {caption_tok [1:]}')
			seq_len = len (caption_tok)
			print (f'seq len - {seq_len}')

			tmp_caption = self.decoder_transform (caption_tok, is_split_into_words=True, return_tensors='pt')
			tok_len = tmp_caption.input_ids.shape [1]

			# print (f'caption ids - {caption.input_ids.shape}')
			# print (f'caption ids - {caption.input_ids}')
			# print (f'decode - {self.decoder_transform.decode (caption.input_ids [0])}')
			print (f'tok len - {tok_len}')

			caption = self.decoder_transform (caption_tok, is_split_into_words=True, max_length=self.max_len, padding='max_length', return_attention_mask=True, return_tensors='pt')

			print (f'ids - {caption.input_ids.shape} - {caption.input_ids}')
			print (f'mask - {caption.attention_mask.shape} -  {caption.attention_mask}')

			caption_src = torch.cat ([caption.input_ids [:, :tok_len], caption.input_ids [:, tok_len+1:]], dim=1)
			caption_src_mask = torch.cat ([caption.attention_mask [:, :tok_len], caption.attention_mask [:, tok_len+1:]], dim=1)

			print (f'src ids - {caption_src.shape} - {caption_src}')
			print (f'src mask - {caption_src_mask.shape} {caption_src_mask}')

			caption_tgt = caption.input_ids [:, 1:]
			caption_tgt_mask = caption.attention_mask [:, 1:]
			
			print (f'tgt ids - {caption_tgt.shape} - {caption_tgt}')
			print (f'tgt mask - {caption_tgt_mask.shape} - {caption_tgt_mask}')

			return image, caption_src, caption_src_mask, caption_tgt, caption_tgt_mask	

		if self.text_transform:
			caption = self.text_transform (f"start {caption_str}", self.word_to_index)
			target = self.text_transform (f"{caption_str} end", self.word_to_index)
			target_seq_len = target.shape [0]
			caption = F.pad (caption, pad=(0, self.max_len-target_seq_len))
			target = F.pad (target, pad=(0, self.max_len-target_seq_len))
		
			return image, caption, target, target_seq_len

# if __name__ == '__main__':
#	 train_dataset = HVGDataset ()
#	 train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)

#	 for _, (image, caption, target, target_seq_len) in enumerate (train_dataloader):
#		 # print (f'caption - {q}')
#		 print (f'image - {image.shape}')
#		 print (f'audio - {audio_file}')
#		 print (f'context - {context_tensor.shape}')
#		 print (f'target - {target.shape}')
#		 print (f'context len - {context_len}')
#		 print (f'target len - {target_len}')
#		 break