import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from random import random

class DataGenerator:
    def __init__(self, image_names, labels, aug_prob=0.0):
        self.__image_names = np.array(image_names)
        self.__labels = np.array(labels)
        self.__aug_prob = aug_prob
        assert len(self.__image_names) == len(self.__labels)
    
    def get_epoch_generator(self, batch_size=32):
        cur_ind = 0
        while True:
            batch_images = None
            y_batch = None
            if cur_ind >= len(self.__image_names):
                break
            
            if cur_ind + batch_size <= len(self.__image_names):
                batch_images = self.__image_names[cur_ind:cur_ind + batch_size]
                y_batch = self.__labels[cur_ind:cur_ind + batch_size]
            else:
                batch_images = self.__image_names[cur_ind:]
                y_batch = self.__labels[cur_ind:]
            
            x_batch = None
            for img_name in batch_images:
                # print(img_name)
                img_path = os.path.join('simple_image_classification\\trainval\\', img_name)
                img_as_array = open_image(img_path, aug_prob=self.__aug_prob)

                if x_batch is None:
                    x_batch = img_as_array
                else:
                    x_batch = torch.cat((x_batch, img_as_array), dim=0)
            cur_ind += len(batch_images)
            y_batch = torch.from_numpy(y_batch)
            # print('x_batch:', x_batch.size(), '\ty_batch:' ,y_batch.size())
            yield x_batch, y_batch

        # мешаем данные каждую эпоху
        ind = np.random.permutation(len(self.__image_names))
        self.__image_names = self.__image_names[ind]
        self.__labels = self.__labels[ind]

def open_image(image_path, aug_prob=0.0):
    img = Image.open(image_path).convert('RGB')
    if random() < aug_prob:
        img_as_array = augment(img)
    else:
        img_as_array = preprocess(img)
    
    assert len(img_as_array.shape) == 4
    assert img_as_array.shape[1] == 3
    #img_as_array = np.rollaxis(img_as_array, 2, 0)
    return img_as_array


def preprocess(img):
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch.unsqueeze(transform_image(img), dim=0)

def augment(img):
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=10, expand=False),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return torch.unsqueeze(transform_image(img), dim=0)