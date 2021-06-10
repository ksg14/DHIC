# Indic library
from json import decoder
import sys
from indicnlp import common
INDIC_NLP_LIB_HOME=r"indic_nlp_library"
INDIC_NLP_RESOURCES=r"indic_nlp_resources"
sys.path.append(r'{}\src'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)

from indicnlp.tokenize import indic_tokenize  

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from transformers import ElectraTokenizer, BertTokenizer

from model.encoder import VitEncoder
from model.decoder import Decoder

from utils.dataset import HVGDataset
from utils.custom_transform import ToSequence

from utils.config import Config

def train (config: Config, encoder: VitEncoder, decoder: Decoder, dataloader: DataLoader):
	for image, caption, caption_mask, target, target_mask in dataloader:
		print (f'image shape - {image.shape}')
		print (f'caption - {caption.shape}')
		print (f'caption mask - {caption_mask.shape}')
		print (f'target - {target.shape}')
		print (f'target mask - {target_mask.shape}')
		# print (f'target_seq_len shape- {target_seq_len.shape}')
		# print (f'target_seq_len - {target_seq_len}')

		# print (f'image[0].shape {image [0].shape}')

		# print (f'max - {image.max ()}')
		# print (f'min - {image.min ()}')

		images_list = [image [i] for i in range (config.batch_sz)]
		# print (type (images_list))
		# print (type (images_list [0]))
		# print (images_list [0].shape)

		enc_last_hidden, enc_attentions = encoder (images_list)

		print (f'vit enc out - {enc_last_hidden.shape}')

		dec_loss, dec_logits, dec_attentions = decoder (config.batch_sz, caption, caption_mask, target, enc_last_hidden)

		print (f'loss - {dec_loss}')

		break

if __name__ == '__main__':
	config = Config ()

	text_transform = ToSequence (tokenizer=indic_tokenize.trivial_tokenize)
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
	train_dataloader = DataLoader (train_dataset, batch_size=config.batch_sz, shuffle=True)

	# Encoder
	encoder = VitEncoder (fe_path=config.pretrained_vitfe_path, vit_path=config.pretrained_vit_path, out_attentions=False)

	# Decoder
	decoder = Decoder (decoder_path=config.pretrained_decoder_path, out_attentions=False)

	train (config=config, \
			encoder=encoder, \
			decoder=decoder, \
			dataloader=train_dataloader)
