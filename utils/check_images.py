import numpy as np
from tqdm import tqdm

from scripts.config import COn

train = np.load (path + 'coco_train_ids.npy')
val = np.load (path + 'coco_dev_ids.npy')
test = np.load (path + 'coco_test_ids.npy')

splits = [train, val, test]
splits_name = ['train', 'val', 'test']

max_len = 0
sum_len = 0
count = 0

for i in range (3):
	count = 0
	missing_list = []
	for img in tqdm (splits [i]):
		try:
			x = np.load (img_path + str (img) + '.npz')
			# print (x ['feat'].shape)
		except:
			count += 1
			missing_list.append (img)
	print ('Missing in ' + splits_name [i] + ' = ' + str (count))
	print (missing_list)

if __name__ == '__main__':
	


