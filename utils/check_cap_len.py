'''
Get caption stats : maxlen and length distribution of caption words.
'''

import json
from os import stat
from typing import List
from tqdm import tqdm
import argparse

from .config import Config

# Indic library
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import sentence_tokenize
from indicnlp.tokenize import indic_tokenize  

def check_in_range (text_range : str, key : int) -> bool :
	'''
	params : '0-50', 23
	output : True if key in range of text_range
	'''
	start, end = text_range.split ('-')
	if end == 'inf':
		if key >= int (start):
			return True
	else:
		if key >= int (start) and key <= int (end):
			return True
	return False

def update_stats (caption : str, stats : dict) -> None:
	tokens = indic_tokenize.trivial_tokenize(caption)
	n_tokens = len (tokens)

	stats ['maxlen'] = max (stats ['maxlen'], n_tokens)
	for key in stats.keys ():
		if key == 'maxlen':
			continue
		if check_in_range (key, n_tokens):
			stats [key] += 1
	return

def check_len_stats (captions : List) -> None:
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
		'501-inf' : 0,
	}
	for img in tqdm (captions):
		update_stats (img ['caption'], stats)
	
	print (stats)
	return

def run_stats_code (train_captions_file : str, val_captions_file : str, test_captions_file : str) -> None:
	with open (train_captions_file, 'r') as file_io:
		train_captions = json.load (file_io)
	with open (val_captions_file, 'r') as file_io:
		val_captions = json.load (file_io)
	with open (test_captions_file, 'r') as file_io:
		test_captions = json.load (file_io)

	print ('Train captions stats - ')
	check_len_stats (train_captions ['annotations'])

	print ('Val captions stats - ')
	check_len_stats (val_captions ['annotations'])

	print ('Test captions stats - ')
	check_len_stats (test_captions ['annotations'])
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get caption len stats')
	parser.add_argument('-r',
                       '--raw',
                       action='store_true',
                       help='get raw caption stats')
	parser.add_argument('-c',
                       '--clean',
                       action='store_true',
                       help='get clean caption stats')

	args = parser.parse_args()

	config = Config ()

	if args.raw:
		print ('Stats for raw dataset captions ...')
		train_captions_file = config.train_captions
		val_captions_file = config.val_captions
		test_captions_file = config.test_captions
		run_stats_code (train_captions_file, val_captions_file, test_captions_file)

	if args.clean:
		print ('Stats for clean dataset captions ...')
		train_captions_file = config.clean_train
		val_captions_file = config.clean_val
		test_captions_file = config.clean_test
		run_stats_code (train_captions_file, val_captions_file, test_captions_file)
	
	
