import json
from tqdm import tqdm

from config import Config

def write_dense_captions (read_file, out_file):
    captions = dict()
    count = 0

    print (f'Reading file {read_file} ...')
    with open (read_file, 'r', encoding='utf8') as file_io:
        for line in file_io:
            columns = line.split ('\t')
            image_id = int (columns [0])
            caption = columns [-1].strip ('\n') + ' |'
            
            if image_id not in captions:
                count += 1
                captions [image_id] = dict()
                captions [image_id] ['caption'] = caption
            else:
                captions [image_id] ['caption'] += ' ' + caption

    print (f'Saving to {out_file} ...')
    with open (out_file, 'w') as file_io:
        json.dump (captions, file_io)
    
    print (count)
    return

if __name__ == '__main__':
    config = Config ()
    files = [ (config.dataset_train_file, config.captions_train_file), (config.dataset_test_file, config.captions_test_file), \
                (config.dataset_dev_file, config.captions_dev_file), (config.dataset_challenge_file, config.captions_challenge_file) ]
    
    for (dataset_file, captions_file) in files:
        write_dense_captions (dataset_file, captions_file)

    print ('Done !')