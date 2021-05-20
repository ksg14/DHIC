import json
from tqdm import tqdm

from config import Config

# Indic library
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

def check_len_stats (captions):
	
	

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
	check_len_stats (train_captions)

	print ('Val captions stats - ')
	check_len_stats (val_captions)

	print ('Test captions stats - ')
	check_len_stats (test_captions)