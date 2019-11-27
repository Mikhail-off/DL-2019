import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import os
from data_generator import DataGenerator
from random import random

class ModelTrainer:
    BEST_GENERATOR = 'best_generator.mdl'
    BEST_DISCRIMINATOR = 'best_discriminator.mdl'
    IMAGE_LOGS = 'images/'
    MODEL_LOGS = 'models/'

    def __init__(self, generator, discriminator, l1_weight, bce_weight, load_from_best=False,
                 D_learn_prob=0.5):
        self.generator = torch.load(self.BEST_GENERATOR) if load_from_best else generator
        self.discriminator = torch.load(self.BEST_DISCRIMINATOR) if load_from_best else discriminator
        self._l1_weight = l1_weight
        self._bce_weight = bce_weight
        self._D_learn_prob = D_learn_prob
        os.makedirs(self.IMAGE_LOGS, exist_ok=True)
        os.makedirs(self.MODEL_LOGS, exist_ok=True)

    def train_epoch(self, opt_G, opt_D, train_data_gen, loss_func_G, loss_func_D, total_size):
        cur_processed = 0
        g_loss_log = [0.]
        d_loss_log = [0.]
        for x_batch, y_batch in train_data_gen:
            out_G = self.generator(x_batch)
            out_D_Fake = self.discriminator(x_batch, out_G)
            out_D_Real = self.discriminator(x_batch, y_batch)

            if random() < self._D_learn_prob:
                opt_D.zero_grad()
                loss = loss_func_D(D_fake=out_D_Fake, D_real=out_D_Real)
                loss.backward()
                opt_D.step()
                d_loss_log.append(loss.cpu().item())
            else:
                opt_G.zero_grad()
                loss = loss_func_G(G_pred=out_G, G_true=y_batch, D_fake=out_D_Fake)
                loss.backward()
                opt_G.step()
                g_loss_log.append(loss.cpu().item())

            cur_processed += x_batch.shape[0]
            print('Image processed %d/%d. G-loss %f, D-loss %f' % (cur_processed, total_size,
                                                                   g_loss_log[-1], d_loss_log[-1]), end='\r')
        print()
        return g_loss_log[1:], d_loss_log[1:]

    def test_model(self, valid_data_gen, loss_func_G, loss_func_D):
        cur_processed = 0
        g_loss_log = []
        d_loss_log = []
        l1_log = []
        l1_loss_f = nn.L1Loss()
        for x_batch, y_batch in valid_data_gen:
            out_G = self.generator(x_batch).detach()
            out_D_Fake = self.discriminator(x_batch, out_G)
            out_D_Real = self.discriminator(x_batch, y_batch)


            d_loss = loss_func_D(D_fake=out_D_Fake, D_real=out_D_Real)
            d_loss_log.append(d_loss.cpu().item())
            g_loss = loss_func_G(G_pred=out_G, G_true=y_batch, D_fake=out_D_Fake)
            g_loss_log.append(g_loss.cpu().item())

            l1_loss = l1_loss_f(out_G, y_batch)
            l1_log.append(l1_loss.cpu().item())

            cur_processed += x_batch.shape[0]
        g_loss_mean = np.mean(g_loss_log)
        d_loss_mean = np.mean(d_loss_log)
        l1_loss_mean = np.mean(l1_log)
        print('Valid G-loss %f, D-loss %f, L1 %f' % (g_loss_mean, d_loss_mean, l1_loss_mean))

        return g_loss_log, d_loss_log

    def sample_model(self, valid_data_gen):
        processed_images = 0
        for x_batch, y_batch in valid_data_gen:
            out_G = self.generator(x_batch)
            resImages = torch.cat([x_batch, y_batch, out_G], dim=2)

            cur_batch_size = x_batch.shape[0]
            image_idx = range(processed_images, processed_images + cur_batch_size)
            image_names = [os.path.join(self.IMAGE_LOGS, '%05d.png' % idx) for idx in image_idx]
            for image_path, img_tensor in zip(image_names, resImages):
                DataGenerator.save_image(image_path, img_tensor.cpu())
            processed_images += cur_batch_size

    def train(self, n_epochs, batch_size, lr_G, lr_D, train_data_epoch_gen, valid_data_epoch_gen, cuda=True,):
        if cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        g_loss_log = []
        d_loss_log = []
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G)
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D)
        loss_D = DiscriminatorLoss(cuda=cuda)
        loss_G = GeneratorLoss(cuda=cuda, l1_weight=self._l1_weight, bce_weight=self._bce_weight)

        best_G_loss = None
        for epoch in range(n_epochs):
            print('Epoch %d/%d\n' % (epoch, n_epochs))
            train_data_gen = train_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda)

            logs = self.train_epoch(opt_G, opt_D, train_data_gen, loss_G, loss_D, len(train_data_epoch_gen))
            g_loss_log.extend(logs[0])
            d_loss_log.extend(logs[1])
            self.test_model(valid_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda),
                            loss_func_D=loss_D, loss_func_G=loss_G)
            if best_G_loss is None or np.mean(logs[0]) < best_G_loss:
                torch.save(self.generator, self.BEST_GENERATOR)
                torch.save(self.discriminator, self.BEST_DISCRIMINATOR)

            generator_model_name = 'generator_epoch%05d.mdl' % epoch
            discriminator_model_name = 'discriminator_epoch%05d.mdl' % epoch
            torch.save(self.generator, os.path.join(self.MODEL_LOGS, generator_model_name))
            torch.save(self.discriminator, os.path.join(self.MODEL_LOGS, discriminator_model_name))

        self.sample_model(valid_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=cuda))

        self.discriminator = self.discriminator.cpu()
        self.generator = self.generator.cpu()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def conv_block(in_ch, out_ch, kernel_size, padding, stride, activation=nn.LeakyReLU(0.2, True), transpose=False,
               use_bn=False, bn_before=True):
    conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
    layers = [conv_layer(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)]
    if use_bn and bn_before:
        layers += [nn.BatchNorm2d(out_ch)]
    layers += [activation]
    if use_bn and not bn_before:
        layers += [nn.BatchNorm2d(out_ch)]
    return layers


class GeneratorP2P(nn.Module):
    def __init__(self, n_blocks=6, filters=24):
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
        layers += conv_block(in_ch, down_ch, kernel_size, padding, stride=2, use_bn=True, bn_before=False)
        #print('Down in: %d\nDown out: %d' % (in_ch, down_ch))
        if sub_block is None:
            sub_block = conv_block(down_ch, 2 * down_ch, kernel_size, padding, stride=1)
        else:
            sub_block = [sub_block]
        layers += sub_block

        output_ch = in_ch if last_filters is None else last_filters
        layers += conv_block(2 * down_ch, output_ch, kernel_size + 1, padding=1, stride=2, transpose=True,
                             use_bn=True, bn_before=False)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        out = self.model(x)
        # print(out.shape)
        # connect over channels
        out_with_skip = torch.cat([x, out], dim=1)
        return out_with_skip

class DiscriminatorP2P(nn.Module):
    def __init__(self, conv_blocks_count=8):
        super().__init__()
        layers = []
        cur_inputs = 3 * 2
        cur_outputs = 0
        filters = 32
        kernel_size = 3
        padding = kernel_size // 2

        layers += conv_block(cur_inputs, filters, kernel_size, padding, stride=1, use_bn=True, bn_before=False)
        cur_inputs = filters
        # stride convs
        for i in range(conv_blocks_count):
            cur_outputs = min(2**i, 8) * filters
            layers += conv_block(cur_inputs, cur_outputs, kernel_size, padding, stride=2, use_bn=True, bn_before=False)
            cur_inputs = cur_outputs

        layers += conv_block(cur_outputs, 1, 1, 0, stride=1, activation=nn.Sigmoid(), use_bn=True, bn_before=True)
        self.model = nn.Sequential(*layers)
        #self.layers = layers


    def forward(self, inp, target):
        x = torch.cat([inp, target], dim=1)
        return self.model(x)


class GeneratorLoss(nn.Module):
    def __init__(self, cuda, l1_weight=1., bce_weight=0.):
        super().__init__()
        self._l1_weight = l1_weight
        self._bce_weight = bce_weight
        self.l1_loss = nn.L1Loss()
        self.BCE_loss = nn.BCELoss()
        self.cuda = cuda

    def __call__(self, G_pred, G_true, D_fake):
        l1_loss = self.l1_loss(G_pred, G_true)
        true_labels = torch.tensor(1.).expand_as(D_fake)
        if self.cuda:
            true_labels = true_labels.cuda()
        bce_loss = self.BCE_loss(D_fake, true_labels)
        loss = self._l1_weight * l1_loss + self._bce_weight * bce_loss
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, cuda):
        super().__init__()
        self.BCE_loss = nn.BCELoss()
        self.cuda = cuda

    def __call__(self, D_fake, D_real):
        labels_for_real = torch.tensor(1.).expand_as(D_real)
        labels_for_fake = torch.tensor(0.).expand_as(D_fake)
        if self.cuda:
            labels_for_fake = labels_for_fake.cuda()
            labels_for_real = labels_for_real.cuda()
        loss_real = self.BCE_loss(D_real, labels_for_real)
        loss_fake = self.BCE_loss(D_fake, labels_for_fake)
        return (loss_real + loss_fake) * 0.5
