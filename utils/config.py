from pathlib import Path
import json
import os

from torch._C import set_flush_denormal

class Config ():
    def __init__ (self, config_path=None) -> None:
        if config_path:
            with open (config_path, 'r') as f:
                config_data = json.load (f)
                self.load_config (**config_data)

        # if not os.path.exists(self.output_path):
        #     os.makedirs(self.output_path)

    
    # data
    data_path = Path ('data')
    captions_path = data_path / 'manual_annotations'
    # captions_path = data_path / 'translated_annotations'
    #raw
    train_captions = captions_path / 'para_captions_train.json'
    val_captions = captions_path / 'para_captions_val.json'
    test_captions = captions_path / 'para_captions_test.json'
    # clean
    clean_train = data_path / 'clean_train.json'
    clean_val = data_path / 'clean_val.json'
    clean_test = data_path / 'clean_test.json'
    # preprocessed
    preprocessed_train = data_path / 'preprocessed_train.json'
    preprocessed_val = data_path / 'preprocessed_val.json'
    preprocessed_test = data_path / 'preprocessed_test.json'
    images_path = Path ('../VG/images')
    word_to_index_path = data_path / 'word_to_index.json'
    index_to_word_path = data_path / 'index_to_word.json'

    # model
    pretrained_model_path = Path ('pretrained_model')
    pretrained_vitfe_path = pretrained_model_path / 'vit_feature_extractor'
    pretrained_vit_path = pretrained_model_path / 'vit'

    max_len = 464

    def save_config (self):
        attributes = [ key for key in Config.__dict__ if key [0] != '_' and not callable(Config.__dict__ [key])]
        save_data = { key : Config.__dict__ [key] for key in attributes }
   
        with open (f'{self.output_path}config.json', 'w') as f:
            json.dump (save_data, f)
        return

    def load_config (self, **kwargs):
        class_attributes = [ key for key in Config.__dict__ if key [0] != '_' and not callable(Config.__dict__ [key])]

        for key, value in kwargs.items():
            if key in class_attributes:
                setattr (Config, key, value)
        # print (Config.__dict__)
        return

