import torch
import numpy
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

class ModelTrainer:
    BEST_MODEL = 'best_model.mdl'

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    def train_epoch(self, opt, batch_size, data_gen, total_size):
        bar = tqdm(total=total_size, leave=True)
        for x_batch, y_batch in data_gen:
            print(x_batch.shape, y_batch.shape)

    def train(self, n_epochs, opt, batch_size, data_epoch_gen, cuda=True):
        if cuda:
            self.discriminator.cuda()
            self.generator.cuda()

        data_gen = data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda)
        self.train_epoch(opt, batch_size, data_gen, len(data_epoch_gen))

        self.discriminator.cpu()
        self.generator.cpu()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class GeneratorP2P(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


class DiscriminatorP2P(nn.Module):
    def __init__(self, conv_blocks_count=8):
        super().__init__()
        layers = []
        cur_inputs = 3
        cur_outputs = 0
        filters = 64
        kernel_size = 3
        padding = kernel_size // 2

        def conv_block(in_ch, out_ch, kernel_size, padding, stride, activation=nn.LeakyReLU(0.2, True)):
            return [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_ch),
                activation
            ]

        layers += conv_block(cur_inputs, filters, kernel_size, padding, stride=1)
        cur_inputs = filters
        # stride convs
        for i in range(conv_blocks_count):
            cur_outputs = min(2**i, 8) * filters
            layers += conv_block(cur_inputs, cur_outputs, kernel_size, padding, stride=2)
            cur_inputs = cur_outputs

        layers += conv_block(cur_outputs, 1, kernel_size, padding, stride=1, activation=nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.layers = layers


    def forward(self, x):
        return self.model(x)