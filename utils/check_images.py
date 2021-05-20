from tqdm import tqdm
import json
import os

from config import Config

def get_image_count (captions, images_path):
	count = 0
	for img in tqdm (captions ['annotations']):
		if os.path.exists (images_path / f"{img ['image_id']}.jpg"):
			count += 1
	return count

if __name__ == '__main__':
	config = Config ()

	with open (config.train_captions, 'r') as file_io:
		train_captions = json.load (file_io)
	with open (config.val_captions, 'r') as file_io:
		val_captions = json.load (file_io)
	with open (config.test_captions, 'r') as file_io:
		test_captions = json.load (file_io)

	images_found = get_image_count (train_captions, config.images_path)
	print (f"{images_found} / {len (train_captions ['annotations'])}")

	images_found = get_image_count (val_captions, config.images_path)
	print (f"{images_found} / {len (val_captions ['annotations'])}")
	
	images_found = get_image_count (test_captions, config.images_path)
	print (f"{images_found} / {len (test_captions ['annotations'])}")
	


