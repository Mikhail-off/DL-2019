from dataset_downloader import  DatasetDownloader
from data_generator import DataGenerator
import os

DATASET_NAMES = ['facades', 'cityscapes']

DATA_PATH = 'data/'
TRAIN_FOLDER = 'train'
VAL_FOLDER = 'val'

def main():
    dataset_downloader = DatasetDownloader(DATASET_NAMES, DATA_PATH)
    dataset_downloader.download()

    facades_train = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], TRAIN_FOLDER))
    facades_val = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], VAL_FOLDER))

if __name__ == '__main__':
    main()