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


HIDDEN_SIZE = 160
BATCH_SIZE = 4
SEQUENCE_LEN = 16
N_LAYERS = 1

IS_CUDA = False

def predict(filename, model, index2word, sent2matrix):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sent = line.split()
            vecotorized = sent2matrix(sent).unsqueeze(0)
            if model.is_cuda:
                model = model.cuda()
                vecotorized = vecotorized.cuda()
            output = model(vecotorized)[0].cpu()
            _, pred = torch.max(output, dim=0)
            translation = []
            for ind in pred:
                index = int(ind)
                translation.append(index2word[index])
            for word, word_tr in zip(sent, translation):
                print(word, word_tr)
            print('-------------------------')
    model = model.cpu()

def main():
    train_generator = DataGenerator(TRAIN_FILE_DE, TRAIN_FILE_EN, BATCH_SIZE, SEQUENCE_LEN)
    valid_generator = DataGenerator(VALIDATION_FILE_DE, VALIDATION_FILE_EN, 1, SEQUENCE_LEN,
                                    train_generator.data, train_generator.target)

    trainer = ModelTrainer(train_generator, valid_generator)
    model = TranslationModel(len(train_generator.data.word2index), len(train_generator.target.word2index), HIDDEN_SIZE,
                             N_LAYERS, is_cuda=IS_CUDA)
    trainer.set_model(model)
    trainer.train(n_epochs=10, cuda=IS_CUDA)
    predict(TRAIN_FILE_DE, model, train_generator.target.index2word,  train_generator.data.sentence2vector)

if __name__ == '__main__':
    main()