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

def check_in_range (text_range, key):
	start, end = text_range.split ('-')
	if key >= int (start) and key <= int (end):
		return True
	return False

def update_stats (caption, stats):
	tokens = indic_tokenize.trivial_tokenize(caption)
	n_tokens = len (tokens)

	stats ['maxlen'] = max (stats ['maxlen'], n_tokens)
	for key in stats.keys ():
		if key == 'max_len':
			continue
		if check_in_range (key, n_tokens):
			stats [key] += 1
	return

def check_len_stats (captions):
	stats = {
		'maxlen' : 0,
		'0-50' : 0,
		'51-100' : 0,
		'101-150' : 0,
		'151-200' : 0,
		'201-250' : 0,
		'251-300' : 0,
		'301-350' : 0,
		'351-400' : 0,
		'401-450' : 0,
		'451-500' : 0,
		'501-' : 0,
	}
	for img in tqdm (captions):
		update_stats (img ['caption'], stats)
	
	print (stats)
	return


if __name__ == '__main__':
	config = Config ()

	with open (config.train_captions, 'r') as file_io:
		train_captions = json.load (file_io)
	with open (config.val_captions, 'r') as file_io:
		val_captions = json.load (file_io)
	with open (config.test_captions, 'r') as file_io:
		test_captions = json.load (file_io)

	print ('Train captions stats - ')
	check_len_stats (train_captions ['annotations'])

	print ('Val captions stats - ')
	check_len_stats (val_captions ['annotations'])

	print ('Test captions stats - ')
	check_len_stats (test_captions ['annotations'])