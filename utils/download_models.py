import os
from transformers import ViTFeatureExtractor, ViTModel

from config import Config

def save_pretrained_model (config):
	try:

		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
		model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
	
		feature_extractor.save_pretrained(config.pretrained_vitfe_path)
		model.save_pretrained(config.pretrained_vit_path)
	except Exception as e:
		print (f'Error - {str (e)}')
		return 1
	return 0

if __name__ == '__main__' :
	config = Config ()

	if not os.path.exists (config.pretrained_model_path):
		os.mkdir (config.pretrained_model_path)
	
	if not os.path.exists (config.pretrained_vitfe_path):
		os.mkdir (config.pretrained_vitfe_path)
	
	if not os.path.exists (config.pretrained_vit_path):
		os.mkdir (config.pretrained_vit_path)
	
	save_pretrained_model (config)
	print ('Done!')
