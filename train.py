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

from transformers import ViTFeatureExtractor, ViTModel, ElectraTokenizer, ElectraModel

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

def train (feature_extractor, model, decoder, dataloader):
	for image, caption, target, target_seq_len in train_dataloader:
	# print (f'image shape - {image.shape}')
	# print (f'caption - {caption.shape}')
	# print (f'target - {target.shape}')
	# print (f'target_seq_len shape- {target_seq_len.shape}')
	# print (f'target_seq_len - {target_seq_len}')

	# print (f'image[0].shape {image [0].shape}')

	# print (f'max - {image.max ()}')
	# print (f'min - {image.min ()}')

	images_list = [image [i] for i in range (config.batch_sz)]
	# print (type (images_list))
	# print (type (images_list [0]))
	# print (images_list [0].shape)

	inputs = feature_extractor(images=images_list, return_tensors="pt")
	outputs = model(**inputs, output_attentions=False, output_hidden_states=False)
	last_hidden_states = outputs.last_hidden_state

	print (f'output shape - {last_hidden_states.shape}')
	break


if __name__ == '__main__':
	config = Config ()

	text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)
	image_transform = T.Compose ([T.ToTensor(), T.Resize ((224, 224))])

	train_dataset = HVGDataset (config.train_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, config.max_len, text_transform=text_transform, image_transform=image_transform)
	train_dataloader = DataLoader (train_dataset, batch_size=config.batch_sz, shuffle=True)

	feature_extractor = ViTFeatureExtractor.from_pretrained(config.pretrained_vitfe_path)
	vit_model = ViTModel.from_pretrained(config.pretrained_vit_path)

	tokenizer = ElectraTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
	electra_model = ElectraModel.from_pretrained("monsoon-nlp/hindi-bert", is_decoder = True)

	train (feature_extractor=feature_extractor, \
			model=model, \
			decoder=decoder, \
			dataloader=train_dataloader)
