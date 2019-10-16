import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from math import floor

from data_generator import DataGenerator
from trainer import Flatten, ModelTrainer

TEST_DIR_NAME = 'simple_image_classification\\test\\'
MARKUP_FILE_NAME = 'simple_image_classification\\labels_trainval.csv'

TRAIN_RATIO = 0.7
TEST_RATIO = 1 - TRAIN_RATIO
INPUT_SIZE = 64 * 64 * 3
NUM_CLASSES = 200

def build_final_model():
    hidden = 100

    model = nn.Sequential(
        Flatten(),
        nn.Linear(INPUT_SIZE, hidden),
        nn.Sigmoid(),
        nn.Linear(hidden, hidden),
        nn.Sigmoid(),
        nn.Linear(hidden, NUM_CLASSES),
        nn.LogSoftmax(dim=-1),
        )
    
    return model


def main():
    df = pd.read_csv(MARKUP_FILE_NAME)
    # кол-во классов
    # print(len(set(list(df['Category'].values))))
    border = floor(len(df) * TRAIN_RATIO)
    df_train = df[:border]
    df_test = df[border: 2*border]
    
    train_gen = DataGenerator(image_names=df_train['Id'], labels=df_train['Category'])
    test_gen = DataGenerator(image_names=df_test['Id'], labels=df_test['Category'])

    trainer = ModelTrainer(train_gen, test_gen)
    model = build_final_model()
    trainer.set_model(model)
    trainer.train(10)

if __name__ == '__main__':
    main()