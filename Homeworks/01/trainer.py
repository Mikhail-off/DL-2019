import torch

class ModelTrainer:
    def __init__(self):
        self.__model = None

    def set_model(self, model):
        self.__model = model
    
    def train(self, epoch, batch_size):
        pass

    