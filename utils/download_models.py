import os
import argparse

from transformers import ViTFeatureExtractor, ViTModel, ElectraTokenizer, ElectraModel

from .config import Config

def save_vit_model (config: Config) -> int:
	try:
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
		model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
	
		feature_extractor.save_pretrained(config.pretrained_vitfe_path)
		model.save_pretrained(config.pretrained_vit_path)
	except Exception as e:
		print (f'Error - {str (e)}')
		return 1
	return 0

def save_electra_decoder (config: Config) -> int:
	try:
		tokenizer = ElectraTokenizer.from_pretrained("monsoon-nlp/hindi-bert")
		model = ElectraModel.from_pretrained("monsoon-nlp/hindi-bert", is_decoder = True)

		tokenizer.save_pretrained(config.pretrained_tokenizer_path)
		model.save_pretrained(config.pretrained_electra_path)
	except Exception as e:
		print (f'Error - {str (e)}')
		return 1
	return 0

if __name__ == '__main__' :
	parser = argparse.ArgumentParser(description='Get caption len stats')
	parser.add_argument('-e',
                       '--encoder',
                       action='store_true',
                       help='get pretrained encoder')
	parser.add_argument('-d',
                       '--decoder',
                       action='store_true',
                       help='get pretrained decoder')
	
	args = parser.parse_args()

	config = Config ()
	
	if args.encoder:
		print (f'Saving encoder model')

		if not os.path.exists (config.pretrained_model_path):
			os.mkdir (config.pretrained_model_path)
		
		if not os.path.exists (config.pretrained_vitfe_path):
			os.mkdir (config.pretrained_vitfe_path)
		
		if not os.path.exists (config.pretrained_vit_path):
			os.mkdir (config.pretrained_vit_path)
	
		save_vit_model (config)
	
	if args.decoder:
		print (f'Saving decoder model')

		if not os.path.exists (config.pretrained_tokenizer_path):
			os.mkdir (config.pretrained_tokenizer_path)
		
		if not os.path.exists (config.pretrained_electra_path):
			os.mkdir (config.pretrained_electra_path)
		
		save_electra_decoder (config)

	print ('Done!')
