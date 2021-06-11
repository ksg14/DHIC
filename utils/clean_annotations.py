import json
from typing import Dict, List
from tqdm import tqdm
import argparse

from config import Config

# Indic library
import sys, os
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import indic_tokenize  

def clean_captions (captions : List, token_threshold: int) -> Dict:
	clean_captions = { 'annotations' : [] }
	for img in tqdm (captions):
		tokens = indic_tokenize.trivial_tokenize (img ['caption'])
		image_file = os.path.join (config.images_path, f"{img ['image_id']}.jpg")

		if len (tokens) >= token_threshold and os.path.exists (image_file):
			clean_captions ['annotations'].append ({
				'image_id' : img ['image_id'],
				'caption' : img ['caption']
			})
	return clean_captions

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get clean captions')
	parser.add_argument('--threshold', type=int, default=20)
		
	args = parser.parse_args()

	config = Config ()

	with open (config.train_captions, 'r') as file_io:
		train_captions = json.load (file_io)
	with open (config.val_captions, 'r') as file_io:
		val_captions = json.load (file_io)
	with open (config.test_captions, 'r') as file_io:
		test_captions = json.load (file_io)

	clean_train = clean_captions (train_captions ['annotations'], args.threshold)

	clean_val = clean_captions (val_captions ['annotations'], args.threshold)

	clean_test = clean_captions (test_captions ['annotations'], args.threshold)

	print (f"Train - {len (clean_train ['annotations'])}")
	print (f"Val - {len (clean_val ['annotations'])}")
	print (f"Test - {len (clean_test ['annotations'])}")

	with open (config.clean_train_captions, 'w') as file_io:
		json.dump (clean_train, file_io)
	with open (config.clean_val_captions, 'w') as file_io:
		json.dump (clean_val, file_io)
	with open (config.clean_test_captions, 'w') as file_io:
		json.dump (clean_test, file_io)
	
	print ('Done !')

