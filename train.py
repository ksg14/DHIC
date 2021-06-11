# Indic library
from pathlib import Path
from typing import Dict, Tuple, Union
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
from torch.optim import Adam

from nltk.translate.bleu_score import sentence_bleu

from transformers import ElectraTokenizer, BertTokenizer

from model.encoder import VitEncoder
from model.decoder import Decoder

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

# import warnings
# warnings.filterwarnings('ignore')

def save_model (model: Union [VitEncoder, Decoder], model_path: Path) -> None:
	try:
		torch.save(model.state_dict(), model_path)
		print (f'Model saved to {model_path}')
	except Exception:
		print (f'unable to save model {str (Exception)}')
	return

def validate (args: argparse.Namespace, config: Config, encoder: VitEncoder, decoder: Decoder, val_dataloader: DataLoader, device: Device) -> float:
	n_len = len (val_dataloader)
	val_loss = 0.0

	encoder.eval ()
	decoder.eval ()

	with torch.no_grad():
		with tqdm(val_dataloader) as tepoch:
			for image, caption, caption_mask, target, target_mask in tepoch:
				tepoch.set_description (f'Validating ')

				image, caption, caption_mask, target, target_mask = image.to (device), caption.to (device), caption_mask.to (device), target.to (device), target_mask.to (device)

				images_list = [image [i] for i in range (args.batch_sz)]

				enc_last_hidden, enc_attentions = encoder (images_list)

				dec_loss, dec_logits, dec_attentions = decoder (args.batch_sz, caption, caption_mask, target, enc_last_hidden)

				val_loss += dec_loss.item ()
				
				tepoch.set_postfix (val_loss=(val_loss / n_len))
				# break
		val_loss /= n_len
	return val_loss

def train (args: argparse.Namespace, config: Config, encoder: VitEncoder, decoder: Decoder, enc_optim: Optimizer, dec_optim: Optimizer, train_dataloader: DataLoader, val_dataloader: DataLoader, device: Device) -> Tuple [Dict, int]:
	epoch_stats = { 'train' : {'loss' : []}, 'val' : {'loss' : [], 'bleu' : [], 'bleu_1' : [], 'bleu_2' : [], 'bleu_3' : [], 'bleu_4' : []} }
	n_len = len (train_dataloader)
	best_epoch_loss = float ('inf')
	best_epoch = -1

	for epoch in range (args.epochs):
		epoch_stats ['train']['loss'].append (0.0)
		encoder.train ()
		decoder.train ()

		with tqdm(train_dataloader) as tepoch:
			for image, caption, caption_mask, target, target_mask in tepoch:
				tepoch.set_description (f'Epoch {epoch}')

				image, caption, caption_mask, target, target_mask = image.to (device), caption.to (device), caption_mask.to (device), target.to (device), target_mask.to (device)

				enc_optim.zero_grad()
				dec_optim.zero_grad()

				if args.logs:
					print (f'image shape - {image.shape}')
					print (f'caption - {caption.shape}')
					print (f'caption mask - {caption_mask.shape}')
					print (f'target - {target.shape}')
					print (f'target mask - {target_mask.shape}')
					# print (f'target_seq_len shape- {target_seq_len.shape}')
					# print (f'target_seq_len - {target_seq_len}')
					# print (f'image[0].shape {image [0].shape}')
					print (f'max - {image.max ()}')
					print (f'min - {image.min ()}')

				images_list = [image [i] for i in range (args.batch_sz)]

				if args.logs:
					print (f'type image list - {type (images_list)}')
					print (f'type image [0] - {type (images_list [0])}')
					print (f'images [0] - {images_list [0].shape}')

				enc_last_hidden, enc_attentions = encoder (images_list)

				if args.logs:
					print (f'vit enc out - {enc_last_hidden.shape}')

				dec_loss, dec_logits, dec_attentions = decoder (args.batch_sz, caption, caption_mask, target, enc_last_hidden)

				dec_loss.backward()
				
				enc_optim.step()
				dec_optim.step()

				with torch.no_grad():
					epoch_stats ['train']['loss'] [-1] += (dec_loss.item () / n_len)
				
				tepoch.set_postfix (train_loss=epoch_stats ['train']['loss'] [-1])
				# break
		# break
		
		val_loss = validate (args=args, config=config, encoder=encoder, \
								decoder=decoder, val_dataloader=val_dataloader, device=device)
		epoch_stats ['val']['loss'].append (val_loss)

		# Save best epoch model
		if val_loss < best_epoch_loss:
			best_epoch_loss = val_loss
			best_epoch = epoch

			print ('Saving new best model')
			save_model (encoder, config.enc_model_path)
			save_model (decoder, config.dec_model_path)
		
		# Save last epoch model
		if epoch == args.epochs-1:
			print ('Saving last epoch model')
			save_model (encoder, config.output_path / 'last_enc_model.pth')
			save_model (decoder, config.output_path / 'last_dec_model.pth')

	return epoch_stats, best_epoch

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Get caption len stats')
	parser.add_argument('-l',
						'--logs',
						action='store_true',
						help='print logs')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch_sz', type=int, default=128)
	parser.add_argument('--lr', type=float, default=1e-6)
	parser.add_argument('--device', type=str, default='cpu')

	args = parser.parse_args()

	config = Config ()

	if torch.cuda.is_available():
		print ('Cuda is available!')
	
	device = torch.device(args.device)
	print(f'Device - {device}')

	text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)
	# image_transform = T.Compose ([T.ToTensor(), T.Resize ((224, 224)), T.Normalize (config.img_mean, config.img_std)])
	image_transform = T.Compose ([T.ToTensor()])
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
	# print (f'0 -  {tokenizer.decode (0)}')
	# print (f'1 -  {tokenizer.decode (1)}')
	# print (f'2 -  {tokenizer.decode (2)}')
	# print (f'3 -  {tokenizer.decode (3)}')
	# print (f'4 -  {tokenizer.decode (4)}')
	# print (f'5 -  {tokenizer.decode (5)}')
	# print (f'6 -  {tokenizer.decode (6)}')
	# print (f'7 -  {tokenizer.decode (7)}')
	# tokenizer.bos_token = '[START]'
	# tokenizer.eos_token = '[END]'

	train_dataset = HVGDataset (config.train_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, config.max_len, text_transform=None, tokenizer=indic_tokenize.trivial_tokenize, decoder_transform=tokenizer, image_transform=image_transform)
	train_dataloader = DataLoader (train_dataset, batch_size=args.batch_sz, shuffle=True)

	val_dataset = HVGDataset (config.val_captions, config.word_to_index_path, config.index_to_word_path, config.images_path, config.max_len, text_transform=None, tokenizer=indic_tokenize.trivial_tokenize, decoder_transform=tokenizer, image_transform=image_transform)
	val_dataloader = DataLoader (val_dataset, batch_size=args.batch_sz, shuffle=True)

	# Encoder
	encoder = VitEncoder (fe_path=config.pretrained_vitfe_path, vit_path=config.pretrained_vit_path, out_attentions=False, do_resize=True, do_normalize=True)

	# Decoder
	decoder = Decoder (decoder_path=config.pretrained_decoder_path, out_attentions=False)

	encoder.to (device)
	decoder.to (device)

	enc_optim = Adam(encoder.parameters(), lr=args.lr)
	dec_optim = Adam(decoder.parameters(), lr=args.lr)

	epoch_stats, best_epoch = train (args=args, \
									config=config, \
									encoder=encoder, \
									decoder=decoder, \
									enc_optim=enc_optim, \
									dec_optim=dec_optim, \
									train_dataloader=train_dataloader, \
									val_dataloader=val_dataloader, \
									device=device)
	
	print (f'Best epoch - {best_epoch} !')

	try:
		with open (config.stats_json_path, 'w') as file_io:
			json.dump (epoch_stats, file_io)
			print (f'Stats saved to {config.stats_json_path}')
	except Exception:
		pickle.dump(epoch_stats, open(config.stats_pkl_path, 'wb'))
		print (f'Stats saved to {config.stats_pkl_path}')
		
	try:
		config.save_config ()
		print (f'Saved config')
	except Exception as e:
		print (f'Unable to save config {str (e)}')
		
		
	print ('Done !')

	
