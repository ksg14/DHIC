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

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

if __name__ == '__main__':
	config = Config ()

	text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)

	train_dataset = HVGDataset (config.train_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, text_transform=text_transform)
	train_dataloader = DataLoader (train_dataset, batch_size=1, shuffle=False)

	for _, (image, caption, target, target_seq_len) in enumerate (train_dataloader):
		print (f'image - {image}.shape')
		print (f'caption - {caption}.shape')
		print (f'target - {target}.shape')
		print (f'target_seq_len - {len (target_seq_len)}, target [0] - {len (target_seq_len [0])}')

		break
	