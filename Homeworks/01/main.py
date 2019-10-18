import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from math import floor
import torchvision.models as models

from data_generator import DataGenerator, open_image
from trainer import Flatten, ModelTrainer


TEST_DIR_NAME = 'simple_image_classification\\test\\'
MARKUP_FILE_NAME = 'simple_image_classification\\labels_trainval.csv'
FINAL_FILE_NAME = 'labels_test.csv'
FINAL_MODEL_PATH = 'models\\final_model.mdl'

TRAIN_RATIO = 0.9
TEST_RATIO = 1 - TRAIN_RATIO
INPUT_SIZE = 224 * 224 * 3
IMAGE_SIZE = (224, 224)
CHANNELS = 3
NUM_CLASSES = 200
EPOCHS = 25
MAX_LEN = 10**5

def build_final_model():
    model_mn = models.mobilenet_v2(pretrained=False)
    model_mn.classifier[1] = nn.Linear(in_features=model_mn.last_channel, out_features=NUM_CLASSES)
    model = nn.Sequential(
        model_mn,
        nn.LogSoftmax(dim=-1),
        )
    
    return model

def predict(model, image_folder=TEST_DIR_NAME, cuda=True):
    image_names = os.listdir(image_folder)
    df = pd.DataFrame(columns=['Id', 'Category'])

    if cuda:
        model.cuda()

    for img_name in sorted(image_names):
        print('Processing', img_name, end='\r')
        img_path = os.path.join(image_folder, img_name)
        img_as_array = open_image(img_path)
        data = img_as_array
        if cuda:
            data = data.cuda()
        result = model(data)
        pred = '%04d' % torch.max(result, 1)[1].cpu().item()
        pred_df = pd.DataFrame(columns=['Id', 'Category'], data=[[img_name, pred]])
        df = df.append(pred_df)
    df.to_csv(FINAL_FILE_NAME, index=False)


def main():
    df = pd.read_csv(MARKUP_FILE_NAME)
    df = df.iloc[np.random.permutation(len(df))]
    if len(df) > MAX_LEN:
        df = df[:MAX_LEN]
        print('Обрезаем выборку до', len(df))
        
    # кол-во классов
    # print(len(set(list(df['Category'].values))))
    border = floor(len(df) * TRAIN_RATIO)
    df_train = df[:border]
    df_test = df[border:]
    
    train_gen = DataGenerator(image_names=df_train['Id'], labels=df_train['Category'], aug_prob=0.3)
    test_gen = DataGenerator(image_names=df_test['Id'], labels=df_test['Category'])

    trainer = ModelTrainer(train_gen, test_gen)
    
    if os.path.exists(FINAL_MODEL_PATH) and os.path.isfile(FINAL_MODEL_PATH):
        model = torch.load(FINAL_MODEL_PATH)
    else:
        model = build_final_model()
    trainer.set_model(model)
    trainer.train(n_epochs=EPOCHS, lr=3e-4, batch_size=96, cuda=True)
    model = trainer.get_model()
    torch.save(model, FINAL_MODEL_PATH)
    predict(model)

if __name__ == '__main__':
    main()