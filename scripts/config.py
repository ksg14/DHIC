
class Config ():
    def __init__ (self):
        # make data folder
        pass
    
    # dataset
    dataset_path = 'hvg_dataset'
    dataset_train_file = f'{dataset_path}/hindi-visual-genome-train.txt'
    dataset_test_file = f'{dataset_path}/hindi-visual-genome-test.txt'
    dataset_dev_file = f'{dataset_path}/hindi-visual-genome-dev.txt'
    dataset_challenge_file = f'{dataset_path}/hindi-visual-genome-challenge-test-set.txt'
    
    # data
    data_path = 'data'
    captions_train_file = f'{data_path}/captions_train.json'
    captions_test_file = f'{data_path}/captions_test.json'
    captions_dev_file = f'{data_path}/captions_dev.json'
    captions_challenge_file = f'{data_path}/captions_challenge.json'
    