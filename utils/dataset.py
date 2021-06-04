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
	def __init__ (self, captions_file : Path, word_to_index_file : Path, index_to_word_file : Path, images_path : Path, max_len : int, text_transform : Callable=None, image_transform : Callable=None) -> None:
		with open (captions_file, 'r') as file_io:
			self.captions = json.load (file_io)
		
		with open (word_to_index_file, 'r') as file_io:
			self.word_to_index = json.load (file_io)
		
		with open (index_to_word_file, 'r') as file_io:
			self.index_to_word = json.load (file_io)
		
		self.max_len = max_len
		self.images_path = images_path
		self.text_transform = text_transform
		self.image_transform = image_transform

	def __len__ (self) -> int:
		return len (self.captions)
		
	def __getitem__ (self, idx: str) -> Tuple:
		image_id = self.captions ['annotations'] [idx] ['image_id']
		caption_str = self.captions ['annotations'] [idx] ['caption']

		# Image
		# image_file = os.path.join (self.images_path, f'{image_id}.jpg')
		image_file = os.path.join (self.images_path, f'16.jpg')
		image = Image.open (image_file)
		if self.image_transform:
			image = self.image_transform (image)

		# Target Caption
		if self.text_transform:
			caption = self.text_transform (f"start {caption_str}", self.word_to_index)
		
		if self.text_transform:
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