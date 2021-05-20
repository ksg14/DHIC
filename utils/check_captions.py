import json
from os import stat
from tqdm import tqdm

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

def has_english_char (text):
	for char in range (len (text)):			
		if char.isalpha ():
			return False
	return True

def clean_captions (captions):
	clean_captions = []
	for img in tqdm (captions):
		sentences=sentence_tokenize.sentence_split(img ['caption'], lang='hi')

		uniq_sentences = set (sentences)
		final_sentences = []
		for sent in uniq_sentences:
			if not has_english_char (sent):
				final_sentences.append (sent)
		clean_captions.append ({
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

	print ('Train captions stats - ')
	clean_train = clean_captions (train_captions ['annotations'])

	print ('Val captions stats - ')
	clean_val = clean_captions (val_captions ['annotations'])

	print ('Test captions stats - ')
	clean_test = clean_captions (test_captions ['annotations'])

	with open (config.clean_train, 'r') as file_io:
		json.dump (clean_train, file_io)
	with open (config.clean_val, 'r') as file_io:
		json.dump (clean_val, file_io)
	with open (config.clean_test, 'r') as file_io:
		json.dump (clean_test, file_io)

