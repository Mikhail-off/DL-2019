import torch
import numpy
import torch.nn as nn
from torchvision import transforms
import os
from data_generator import DataGenerator

class ModelTrainer:
    BEST_MODEL = 'best_model.mdl'
    IMAGE_LOGS = 'images/'

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        os.makedirs(self.IMAGE_LOGS, exist_ok=True)

    def train_epoch(self, opt_G, opt_D, train_data_gen, total_size, loss_func):
        cur_processed = 0
        loss_log = []
        for x_batch, y_batch in train_data_gen:
            out_G = self.generator(x_batch)
            out_D = self.discriminator(out_G)

            opt_G.zero_grad()
            loss = loss_func(out_G, y_batch)
            loss.backward()
            opt_G.step()

            loss = loss.cpu().item()
            loss_log.append(loss)
            cur_processed += x_batch.shape[0]
            print('Image processed %d/%d. Loss %f' % (cur_processed, total_size, loss), end='\r')

        return loss_log

    def test_model(self, valid_data_gen):
        processed_images = 0
        for x_batch, y_batch in valid_data_gen:
            out_G = self.generator(x_batch)
            resImages = torch.cat([x_batch, y_batch, out_G], dim=1)
            cur_batch_size = x_batch.shape[0]
            image_idx = range(processed_images, processed_images + cur_batch_size)
            image_names = [os.path.join(self.IMAGE_LOGS, '%05d.png' % idx) for idx in image_idx]
            for image_path, img_tensor in zip(image_names, resImages):
                DataGenerator.save_image(image_path, img_tensor.cpu())


    def train(self, n_epochs, batch_size, lr_G, lr_D, train_data_epoch_gen, valid_data_opech_gen, cuda=True):
        if cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        loss_log = []
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D)
        for epoch in range(n_epochs):
            print('Epoch %d/%d' % (epoch, n_epochs))
            train_data_gen = train_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda)
            loss = nn.L1Loss()
            loss_log.extend(self.train_epoch(opt_G, opt_D, train_data_gen, len(train_data_epoch_gen), loss))

        self.test_model(valid_data_opech_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda))

        self.discriminator = self.discriminator.cpu()
        self.generator = self.generator.cpu()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def conv_block(in_ch, out_ch, kernel_size, padding, stride, activation=nn.LeakyReLU(0.2, True), transpose=False):
    conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
    return [
        conv_layer(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        activation
    ]


class GeneratorP2P(nn.Module):
    def __init__(self, n_blocks=6, filters=32):
        super().__init__()
        cur_block = None
        cur_filters = 2**(n_blocks - 2) * filters

        for i in range(n_blocks - 2):
            cur_block = UNetBlock(cur_filters, cur_block)
            cur_filters //= 2
        cur_block = UNetBlock(in_ch=3, sub_block=cur_block, first_filters=cur_filters * 2, last_filters=cur_filters)
        self.unet = cur_block
        self.last_conv = nn.Sequential(*conv_block(cur_filters + 3, 3, 3, 1, stride=1, activation=nn.Tanh()))

    def forward(self, x):
        unet_out = self.unet(x)
        return self.last_conv(unet_out)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, sub_block=None, first_filters=None, last_filters=None):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        layers = []

        down_ch = 2 * in_ch if first_filters is None else first_filters
        layers += conv_block(in_ch, down_ch, kernel_size, padding, stride=2)
        #print('Down in: %d\nDown out: %d' % (in_ch, down_ch))
        if sub_block is None:
            sub_block = conv_block(down_ch, 2 * down_ch, kernel_size, padding, stride=1)
        else:
            sub_block = [sub_block]
        layers += sub_block

        output_ch = in_ch if last_filters is None else last_filters
        layers += conv_block(2 * down_ch, output_ch, kernel_size + 1, padding=1, stride=2, transpose=True)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.model(x)
        # print(out.shape)
        # connect over channels
        out_with_skip = torch.cat([x, out], dim=1)
        return  out_with_skip

class DiscriminatorP2P(nn.Module):
    def __init__(self, conv_blocks_count=8):
        super().__init__()
        layers = []
        cur_inputs = 3
        cur_outputs = 0
        filters = 16
        kernel_size = 3
        padding = kernel_size // 2

        layers += conv_block(cur_inputs, filters, kernel_size, padding, stride=1)
        cur_inputs = filters
        # stride convs
        for i in range(conv_blocks_count):
            cur_outputs = min(2**i, 8) * filters
            layers += conv_block(cur_inputs, cur_outputs, kernel_size, padding, stride=2)
            cur_inputs = cur_outputs

        layers += conv_block(cur_outputs, 1, kernel_size, padding, stride=1, activation=nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        #self.layers = layers


    def forward(self, x):
        return self.model(x)

""""
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_loss = nn.L1Loss()

    def __call__(self, y_pred, y_true):
        return self.simple_loss(y_true, )
"""