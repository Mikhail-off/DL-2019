import os
from data_generator import DataGenerator
from net_trainer import TranslationModel, ModelTrainer
import torch
from tqdm import tqdm

DATA_DIR = 'homework_machine_translation_de-en/'
MODELS_DIR = 'models/'
BEST_MODEL_NAME = os.path.join(MODELS_DIR, 'model_best.mdl')


TRAIN_FILE = 'train.de-en'
TRAIN_FILE_DE = os.path.join(DATA_DIR, TRAIN_FILE + '.de')
TRAIN_FILE_EN = os.path.join(DATA_DIR, TRAIN_FILE + '.en')

VALIDATION_FILE = 'val.de-en'
VALIDATION_FILE_DE = os.path.join(DATA_DIR, VALIDATION_FILE + '.de')
VALIDATION_FILE_EN = os.path.join(DATA_DIR, VALIDATION_FILE + '.en')

TEST_FILE_DE = os.path.join(DATA_DIR, 'test1.de-en.de')
TEST_FILE_EN = os.path.join(DATA_DIR, 'test1.de-en.en')

N_EPOCH = 50
HIDDEN_SIZE = 128
BATCH_SIZE = 128
SEQUENCE_LEN = 64
N_LAYERS = 2
LEARNING_RATE = 0.0001

IS_CUDA = True

def predict(filename, target_filename, model, index2word, sent2matrix):
    f_in = open(filename, 'r', encoding='utf-8')
    f_out = open(target_filename, 'w', encoding='utf-8', buffering=4096)
    for line in tqdm(f_in):
        sent = line.split()
        vecotorized = sent2matrix(sent).unsqueeze(0)
        if model.is_cuda:
            model = model.cuda()
            vecotorized = vecotorized.cuda()
        output = model(vecotorized)[0].cpu()
        _, pred = torch.max(output, dim=-1)
        translation = []
        for ind in pred:
            index = int(ind)
            translation.append(index2word[index])
        translation_str = ' '.join(translation) + '\n'
        f_out.write(translation_str)
    model = model.cpu()
    f_in.close()
    f_out.close()

def main():
    train_generator = DataGenerator(TRAIN_FILE_DE, TRAIN_FILE_EN, BATCH_SIZE, SEQUENCE_LEN)
    valid_generator = DataGenerator(VALIDATION_FILE_DE, VALIDATION_FILE_EN, BATCH_SIZE, SEQUENCE_LEN,
                                    train_generator.data, train_generator.target)

    trainer = ModelTrainer(train_generator, valid_generator)
    if os.path.exists(BEST_MODEL_NAME):
        model = torch.load(BEST_MODEL_NAME)
    else:
        model = TranslationModel(len(train_generator.data.word2index), len(train_generator.target.word2index), HIDDEN_SIZE,
                             N_LAYERS, is_cuda=IS_CUDA)
    trainer.set_model(model)
    trainer.train(n_epochs=N_EPOCH, cuda=IS_CUDA, lr=LEARNING_RATE, is_force=False)

    #predict(TRAIN_FILE_DE, TRAIN_FILE_EN + '.txt', model, train_generator.target.index2word,
    #        train_generator.data.sentence2vector)
    #predict(VALIDATION_FILE_DE, VALIDATION_FILE_EN + '.txt', model, train_generator.target.index2word,
    #        train_generator.data.sentence2vector)
    predict(TEST_FILE_DE, TEST_FILE_EN, model, train_generator.target.index2word,  train_generator.data.sentence2vector)

if __name__ == '__main__':
    main()