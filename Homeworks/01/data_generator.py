import torch
import numpy as np
from PIL import Image
import os

class DataGenerator:
    def __init__(self, image_names, labels):
        self.__image_names = np.array(image_names)
        self.__labels = np.array(labels)
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
            
            x_batch = []
            for img_name in batch_images:
                img_path = os.path.join('simple_image_classification\\trainval\\', img_name)
                img = Image.open(img_path).convert('RGB')
                img_as_array = np.array(img)
                assert len(img_as_array.shape) == 3
                assert img_as_array.shape[-1] == 3

                x_batch.append(img_as_array)
            x_batch = np.array(x_batch)
            yield x_batch, y_batch

        # мешаем данные каждую эпоху
        ind = np.random.permutation(len(self.__image_names))
        self.__image_names = self.__image_names[ind]
        self.__labels = self.__labels[ind]
        
        