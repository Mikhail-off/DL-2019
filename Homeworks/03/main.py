from dataset_downloader import  DatasetDownloader
from data_generator import DataGenerator
from net_trainer import ModelTrainer, GeneratorP2P, DiscriminatorP2P
import os
import torch

DATASET_NAMES = ['facades', 'cityscapes']

DATA_PATH = 'data/'
TRAIN_FOLDER = 'train'
VAL_FOLDER = 'val'

N_EPOCHS = 200
LR_G = 1e-3
LR_D = 1e-2
BATCH_SIZE = 16
CUDA = True
L1_WEIGHT = 1.
BCE_WEIGHT = 0.

def main():
    dataset_downloader = DatasetDownloader(DATASET_NAMES, DATA_PATH)
    dataset_downloader.download()

    trainer = ModelTrainer(GeneratorP2P(), DiscriminatorP2P(), l1_weight=L1_WEIGHT, bce_weight=BCE_WEIGHT)

    facades_train = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], TRAIN_FOLDER))
    facades_val = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], VAL_FOLDER))

    trainer.train(n_epochs=N_EPOCHS,
                  batch_size=BATCH_SIZE,
                  lr_G=LR_G,
                  lr_D=LR_D,
                  train_data_epoch_gen=facades_train,
                  valid_data_epoch_gen=facades_val,
                  cuda=CUDA)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    main()