# Indic library
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import indic_tokenize  

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import ViTFeatureExtractor, ViTModel

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

if __name__ == '__main__':
	config = Config ()

	text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)
	image_transform = T.Compose ([T.ToTensor(), torch.squeeze ()])

	train_dataset = HVGDataset (config.train_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, text_transform=text_transform, image_transform=image_transform)
	train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)

	feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
	model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

	for image, caption, target, target_seq_len in train_dataloader:
		print (f'image - {type (image)}')
		print (f'image len - {len (image)}')
		print (f'image [0] type - {type (image [0])}')
		print (f'image [0] shape - {image [0].shape}')
		print (f'caption - {caption.shape}')
		print (f'target - {target.shape}')
		print (f'target_seq_len - {target_seq_len.shape}')

		inputs = feature_extractor(images=image, return_tensors="pt")
		outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
		last_hidden_states = outputs.last_hidden_state

		print (f'output shape - {last_hidden_states.shape}')
		break
	