import os
from data_generator import DataGenerator
from net_trainer import TranslationModel, ModelTrainer
import torch

DATA_DIR = 'homework_machine_translation_de-en/'
MODELS_DIR = 'models/'

TRAIN_FILE = 'train.de-en'
TRAIN_FILE_DE = os.path.join(DATA_DIR, TRAIN_FILE + '.de')
TRAIN_FILE_EN = os.path.join(DATA_DIR, TRAIN_FILE + '.en')

VALIDATION_FILE = 'val.de-en'
VALIDATION_FILE_DE = os.path.join(DATA_DIR, VALIDATION_FILE + '.de')
VALIDATION_FILE_EN = os.path.join(DATA_DIR, VALIDATION_FILE + '.en')

TEST_FILE_DE = os.path.join(DATA_DIR, 'test1.de-en.de')


HIDDEN_SIZE = 16
BATCH_SIZE = 4
SEQUENCE_LEN = 16
N_LAYERS = 1

IS_CUDA = False

def main():
    train_generator = DataGenerator(TRAIN_FILE_DE, TRAIN_FILE_EN, BATCH_SIZE, SEQUENCE_LEN)
    valid_generator = DataGenerator(VALIDATION_FILE_DE, VALIDATION_FILE_EN, BATCH_SIZE, SEQUENCE_LEN)
    print(len(train_generator.data.word2index), len(train_generator.target.word2index))
    trainer = ModelTrainer(train_generator, valid_generator)
    model = TranslationModel(len(train_generator.data.word2index), len(train_generator.target.word2index), HIDDEN_SIZE,
                             N_LAYERS, is_cuda=IS_CUDA)
    trainer.set_model(model)
    trainer.train(n_epochs=1, cuda=IS_CUDA)

if __name__ == '__main__':
    main()