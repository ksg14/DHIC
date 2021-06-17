from pathlib import Path
import sys
from typing import Callable, Dict
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp.tokenize import indic_tokenize  

import numpy as np
from tqdm import tqdm
import json
import pickle
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import BertTokenizer

from config import Config

def find_max_len (captions: Dict, tokenizer: Callable=None, dec_tokenizer: Callable=None) -> int:
	max_len_tok = 0
	max_len_dec_tok = 0

	for cap in tqdm (captions ['annotations']):
		image_id = cap ['image_id']
		caption_str = cap ['caption']

		if tokenizer:
			caption_tok = tokenizer (caption_str)
		else:
			caption_tok = dec_tokenizer.tokenize (caption_str)

		max_len_tok = max (max_len_tok, len (caption_tok))

		tmp_caption = dec_tokenizer (caption_tok, is_split_into_words=True, return_tensors='pt')
		dec_tok_len = tmp_caption.input_ids.shape [1]

		max_len_dec_tok = max (max_len_dec_tok, dec_tok_len)
	
	return max_len_tok, max_len_dec_tok

if __name__ == '__main__':
	config = Config ()

	dec_tokenizer = BertTokenizer.from_pretrained(config.pretrained_tokenizer_path)
	# indic_tokenize.trivial_tokenize

	with open (config.clean_train_captions, 'r') as file_io:
		train = json.load (file_io)
	with open (config.clean_val_captions, 'r') as file_io:
		val = json.load (file_io)
	with open (config.clean_test_captions, 'r') as file_io:
		test = json.load (file_io)

	train_stats = find_max_len (train, indic_tokenize.trivial_tokenize, dec_tokenizer)
	val_stats = find_max_len (val, indic_tokenize.trivial_tokenize, dec_tokenizer)
	test_stats = find_max_len (test, indic_tokenize.trivial_tokenize, dec_tokenizer)
	
	print (f'Max len for train - pre dec : {train_stats [0]}  after dec : {train_stats [1]}')
	print (f'Max len for val - pre dec : {val_stats [0]}  after dec : {val_stats [1]}')
	print (f'Max len for test - pre dec : {test_stats [0]}  after dec : {test_stats [1]}')



	