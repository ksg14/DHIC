import json
from typing import List
from tqdm import tqdm
import re

from config import Config

# Indic library
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import sentence_tokenize
from indicnlp.tokenize import indic_tokenize  

def has_english_char (text : str) -> bool:
	reg = re.compile(r'[a-zA-Z]')		
	for char in text:	
		if reg.match(char):
			# print (f'char - {char}')
			return True
	return False

def clean_captions (captions : List) -> List:
	clean_captions = { 'annotations' : [] }
	for img in tqdm (captions):
		sentences=sentence_tokenize.sentence_split(img ['caption'], lang='hi')

		# print (f'raw - {sentences} , len - {len (sentences)}')

		uniq_sentences = set (sentences)
		
		# print (f'uniq - {uniq_sentences}  , len - {len (uniq_sentences)}')

		final_sentences = []
		for sent in uniq_sentences:
			# print (f'sent - {sent}')
			if not has_english_char (sent):
				# print ('Added')
				final_sentences.append (sent)
			# else:
			# 	print ('not added')

		clean_captions ['annotations'].append ({
			'image_id' : img ['image_id'],
			'caption' : ' '.join (final_sentences)
		})
	return clean_captions

if __name__ == '__main__':
	config = Config ()

	with open (config.train_captions, 'r') as file_io:
		train_captions = json.load (file_io)
	with open (config.val_captions, 'r') as file_io:
		val_captions = json.load (file_io)
	with open (config.test_captions, 'r') as file_io:
		test_captions = json.load (file_io)

	clean_train = clean_captions (train_captions ['annotations'])

	clean_val = clean_captions (val_captions ['annotations'])

	clean_test = clean_captions (test_captions ['annotations'])

	with open (config.clean_train, 'w') as file_io:
		json.dump (clean_train, file_io)
	with open (config.clean_val, 'w') as file_io:
		json.dump (clean_val, file_io)
	with open (config.clean_test, 'w') as file_io:
		json.dump (clean_test, file_io)

