from dataset_downloader import  DatasetDownloader
from data_generator import DataGenerator
from net_trainer import ModelTrainer, GeneratorP2P, DiscriminatorP2P
import os
import torch

DATASET_NAMES = ['facades', 'cityscapes']

DATA_PATH = 'data/'
TRAIN_FOLDER = 'train'
VAL_FOLDER = 'val'

N_EPOCHS = 2001
LR_G = 2e-4
LR_D = 2e-4
BATCH_SIZE = 8
CUDA = True
L1_WEIGHT = 100.
BCE_WEIGHT = 1
LOAD = True
#D_LEARN_PROB = 0.5

def main():
    dataset_downloader = DatasetDownloader(DATASET_NAMES, DATA_PATH)
    dataset_downloader.download()

    trainer = ModelTrainer(GeneratorP2P(), DiscriminatorP2P(), l1_weight=L1_WEIGHT, bce_weight=BCE_WEIGHT,
                           load_from_best=LOAD, cuda=CUDA)

    facades_train = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], TRAIN_FOLDER))
    facades_val = DataGenerator(os.path.join(DATA_PATH, DATASET_NAMES[0], VAL_FOLDER))

    trainer.train(n_epochs=N_EPOCHS,
                  batch_size=BATCH_SIZE,
                  lr_G=LR_G,
                  lr_D=LR_D,
                  train_data_epoch_gen=facades_train,
                  valid_data_epoch_gen=facades_val)

if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    main()