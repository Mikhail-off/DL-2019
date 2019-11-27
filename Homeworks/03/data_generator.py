import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os

THREAD_COUNT = 8

class DataGenerator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_names = list(os.listdir(dataset_path))

    def get_epoch_generator(self, batch_size=32, cuda=True):
        assert batch_size != 0
        image_count = len(self.image_names)
        for i in range(0, image_count, batch_size):
            batch_start = i
            batch_end = i + batch_size
            if batch_start + batch_size > image_count:
                batch_end = image_count

            batch_image_paths = map(lambda x: os.path.join(self.dataset_path, x), self.image_names[batch_start:batch_end])

            batch_tensor = list(map(self.open_image, batch_image_paths))
            batch_tensor = torch.stack(batch_tensor)
            if cuda:
                batch_tensor = batch_tensor.cuda()

            assert len(batch_tensor.shape) == 4
            width = batch_tensor.shape[3]
            yield batch_tensor[:, :, :, width // 2:], batch_tensor[:, :, :, 0:width // 2]

    def open_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_as_array = (np.array(image) - 127.5) / 127.5
        tensor = torch.from_numpy(img_as_array)
        return tensor

    @staticmethod
    def deprocess_image(tensor):
        tensor = tensor * 127.5 + 127.5
        transform = transforms.Compose([transforms.ToPILImage('RGB')])
        image = transform(tensor)
        return image

    @staticmethod
    def save_image(image_path, tensor):
        image = DataGenerator.deprocess_image(tensor)
        image.save(image_path)

    def __len__(self):
        return len(self.image_names)
