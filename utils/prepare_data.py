import json
from pathlib import WindowsPath
from typing import List, Tuple
from tqdm import tqdm

from utils.config import Config

# Indic library
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import indic_tokenize  

def get_vocab (corpus : List) -> Tuple:
	word_to_index = dict ()
	index_to_word = dict ()
	
	word_to_index ['pad'] = 0
	word_to_index ['start'] = 1
	word_to_index ['end'] = 2
	word_to_index ['unk'] = 3
	index_to_word [0] = 'pad'
	index_to_word [1] = 'start'
	index_to_word [2] = 'end'
	index_to_word [3] = 'unk'
	start_idx = 4

	for img in tqdm (corpus):
		tokens = indic_tokenize.trivial_tokenize (img ['caption'])

		for tok in tokens:
			if tok not in word_to_index:
				word_to_index [tok] = start_idx
				index_to_word [start_idx] = tok
				start_idx += 1
	return word_to_index, index_to_word

if __name__ == '__main__':
	config = Config

	with open (config.train_captions, 'r') as file_io:
		train_captions = json.load (file_io)
	
	word_to_index, index_to_word = get_vocab (train_captions ['annotations'])

	print (f'Unique tokens - {len (word_to_index)}')

	print ('Saving wtio and itow ...')
	with open (config.word_to_index_path, 'w') as file_io:
		json.dump (word_to_index, file_io)
	
	with open (config.index_to_word_path, 'w') as file_io:
		json.dump (index_to_word, file_io)

	print ('Done !')



