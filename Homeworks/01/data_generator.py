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
                # print(img_name)
                img_path = os.path.join('simple_image_classification\\trainval\\', img_name)
                img_as_array = open_image(img_path)

                x_batch.append(img_as_array)
            x_batch = np.array(x_batch).astype(np.float32)
            cur_ind += len(batch_images)

            yield torch.from_numpy(x_batch), torch.from_numpy(y_batch)

        # мешаем данные каждую эпоху
        ind = np.random.permutation(len(self.__image_names))
        self.__image_names = self.__image_names[ind]
        self.__labels = self.__labels[ind]

def open_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_as_array = np.array(img)
    img_as_array = preprocess(img_as_array)
    assert len(img_as_array.shape) == 3
    assert img_as_array.shape[-1] == 3
    img_as_array = np.rollaxis(img_as_array, 2, 0)
    return img_as_array

# mobile_net
def preprocess(img_as_array):
    return ((img_as_array - 127.5) / 127.5).astype(np.float32)