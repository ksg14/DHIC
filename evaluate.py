# Indic library
from pathlib import Path
from typing import Dict, List, Tuple, Union
from torch.optim.optimizer import Optimizer
from torch.types import Device

import sys
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
from torch.nn import functional as F

from nltk.translate.bleu_score import sentence_bleu

from transformers import ElectraTokenizer, BertTokenizer

from model.encoder import VitEncoder
from model.decoder import Decoder

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

# import warnings
# warnings.filterwarnings('ignore')

def evaluate (args: argparse.Namespace, config: Config, tokenizer: BertTokenizer, encoder: VitEncoder, decoder: Decoder, dataloader: DataLoader, device: Device) -> List:
	n_len = len (dataloader)
	predictions = []

	encoder.eval ()
	decoder.eval ()

	with torch.no_grad():
		with tqdm(dataloader) as tepoch:
			for image_id, caption_str, image, caption, caption_mask, target, target_mask in tepoch:
				tepoch.set_description (f'Evaluating ')

				image, caption, caption_mask, target, target_mask = image, caption.to (device), caption_mask.to (device), target.to (device), target_mask.to (device)

				images_list = [image [i] for i in range (args.batch_sz)]

				enc_last_hidden, enc_attentions = encoder (images_list)

				start_id = torch.tensor (args.batch_sz, 1, device=device)
				dec_out = decoder.generate (input_ids=start_id, enc_last_hidden=enc_last_hidden, strategy=args.strategy, max_len=config.max_len, beams=args.beams)

				pred_caption_str = tokenizer.decode(dec_out, skip_special_tokens=True)

				predictions.append ({
					'image_id' : image_id, 
					'gt_caption' : caption_str [0], 
					'pred_caption' : pred_caption_str
				})
				
				# tepoch.set_postfix (val_loss=(val_loss / n_len))
				# break
	return predictions

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training code')
	parser.add_argument('-l',
						'--logs',
						action='store_true',
						help='print logs')
	parser.add_argument('--config', type=str, default=None)
	parser.add_argument('--batch_sz', type=int, default=1)
	parser.add_argument('--strategy', type=str, default='greedy')
	parser.add_argument('--beams', type=int, default=5)
	parser.add_argument('--device', type=str, default='cpu')

	args = parser.parse_args()

	if args.config:
		config = Config (args.config)
	else:
		config = Config ()

	if torch.cuda.is_available():
		print ('Cuda is available!')
	
	device = torch.device(args.device)
	print(f'Device - {device}')

	# text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)
	# image_transform = T.Compose ([T.ToTensor(), T.Resize ((224, 224)), T.Normalize (config.img_mean, config.img_std)])
	image_transform = T.Compose ([T.ToTensor(), T.Resize ((224, 224))])
	tokenizer = BertTokenizer.from_pretrained(config.pretrained_tokenizer_path)

	# print (f'padding side {tokenizer.padding_side}')
	# print (f'bos tok {tokenizer.bos_token}')
	# print (f'bos tok id {tokenizer.bos_token_id}')
	# print (f'eos tok {tokenizer.eos_token}')
	# print (f'eos tok id {tokenizer.eos_token_id}')
	# print (f'pad tok {tokenizer.pad_token}')
	# print (f'pad tok id {tokenizer.pad_token_id}')
	# print (f'mask tok {tokenizer.mask_token}')
	# print (f'vocab size {tokenizer.vocab_size}')
	print (f'0 -  {tokenizer.decode (0)}')
	print (f'1 -  {tokenizer.decode (1)}')
	print (f'2 -  {tokenizer.decode (2)}')
	# print (f'3 -  {tokenizer.decode (3)}')
	# print (f'4 -  {tokenizer.decode (4)}')
	# print (f'5 -  {tokenizer.decode (5)}')
	# print (f'6 -  {tokenizer.decode (6)}')
	# print (f'7 -  {tokenizer.decode (7)}')
	# print (f'80 -  {tokenizer.decode (80)}')
	# tokenizer.bos_token = '[START]'
	# tokenizer.eos_token = '[END]'

	test_dataset = HVGDataset (config.clean_test_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, config.max_len, text_transform=None, tokenizer=indic_tokenize.trivial_tokenize, decoder_transform=tokenizer, image_transform=image_transform)
	test_dataloader = DataLoader (test_dataset, batch_size=args.batch_sz, shuffle=True)

	# Encoder
	encoder = VitEncoder (fe_path=config.pretrained_vitfe_path, vit_path=config.pretrained_vit_path, device=device, out_attentions=False, do_resize=True, do_normalize=True)

	# Decoder
	decoder = Decoder (decoder_path=config.pretrained_decoder_path, out_attentions=False)

	encoder.to (device)
	decoder.to (device)

	epoch_stats, best_epoch = evaluate (args=args, \
									config=config, \
									tokenizer=tokenizer, \
									encoder=encoder, \
									decoder=decoder, \
									dataloader=test_dataloader, \
									device=device)
	
	with open (config.stats_json_path, 'w') as file_io:
		json.dump (epoch_stats, file_io)

	print ('Done !')

	
