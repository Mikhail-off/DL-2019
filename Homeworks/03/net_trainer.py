import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import os
from data_generator import DataGenerator
from random import random
from time import time

def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad


class ModelTrainer:
    BEST_GENERATOR = 'best_generator.mdl'
    BEST_DISCRIMINATOR = 'best_discriminator.mdl'
    IMAGE_LOGS = 'images/'
    MODEL_LOGS = 'models/'

    def __init__(self, generator, discriminator, l1_weight, bce_weight, load_from_best=False,
                 D_learn_prob=0.5, cuda=True):
        self.generator = torch.load(self.BEST_GENERATOR) if load_from_best else generator
        self.discriminator = torch.load(self.BEST_DISCRIMINATOR) if load_from_best else discriminator
        self._l1_weight = l1_weight
        self._bce_weight = bce_weight
        #self._D_learn_prob = D_learn_prob
        self.cuda = cuda
        os.makedirs(self.IMAGE_LOGS, exist_ok=True)
        os.makedirs(self.MODEL_LOGS, exist_ok=True)

    def train_epoch(self, opt_G, opt_D, train_data_gen, total_size, D_learning):
        cur_processed = 0
        g_loss_log = [0.]
        d_loss_log = [0.]

        for x_batch, y_batch in train_data_gen:
            if D_learning:
                set_requires_grad(self.discriminator, True)
                opt_D.zero_grad()
                loss = self.loss_D(x_batch, y_batch)
                loss.backward()
                opt_D.step()
                d_loss_log.append(loss.cpu().item())
            else:
                set_requires_grad(self.discriminator, False)
                opt_G.zero_grad()
                loss = self.loss_G(x_batch, y_batch)
                loss.backward()
                opt_G.step()
                g_loss_log.append(loss.cpu().item())

            cur_processed += x_batch.shape[0]
            print('Image processed %d/%d. G-loss %f, D-loss %f' % (cur_processed, total_size,
                                                                   g_loss_log[-1], d_loss_log[-1]), end='\r')
        print()
        g_loss_log, d_loss_log = g_loss_log[1:], d_loss_log[1:]
        return g_loss_log, d_loss_log

    def test_model(self, valid_data_gen):
        cur_processed = 0
        g_loss_log = []
        d_loss_log = []
        l1_log = []
        l1_loss_f = nn.L1Loss()

        for x_batch, y_batch in valid_data_gen:
            d_loss = self.loss_D(x_batch, y_batch)
            d_loss_log.append(d_loss.cpu().item())
            g_loss = self.loss_G(x_batch, y_batch)
            g_loss_log.append(g_loss.cpu().item())

            l1_loss = l1_loss_f(self.generator(x_batch), y_batch)
            l1_log.append(l1_loss.cpu().item())

            cur_processed += x_batch.shape[0]
        g_loss_mean = np.mean(g_loss_log)
        d_loss_mean = np.mean(d_loss_log)
        l1_loss_mean = np.mean(l1_log)
        print('Valid G-loss %f, D-loss %f, L1 %f' % (g_loss_mean, d_loss_mean, l1_loss_mean))
        print()
        return g_loss_log, d_loss_log

    def loss_D(self, x_data, y_data):
        G_fake = self.generator(x_data).detach()
        D_fake = self.discriminator(x_data, G_fake)

        fake_labels = torch.zeros_like(D_fake)#.uniform_(0.01, 0.1)

        loss_fake = nn.BCELoss()(D_fake, fake_labels)
        #print(D_fake.view(-1))
        D_real = self.discriminator(x_data, y_data)
        #print(D_real.view(-1))
        real_labels = torch.ones_like(D_real)#.uniform_(0.9, 0.99)

        loss_real = nn.BCELoss()(D_real, real_labels)

        return (loss_fake + loss_real) * 0.5

    def loss_G(self, x_data, y_data):
        G_fake = self.generator(x_data)
        l1_loss = nn.L1Loss()(G_fake, y_data)

        D_fake = self.discriminator(x_data, G_fake)
        fake_labels = torch.ones_like(D_fake)
        if self.cuda:
            fake_labels = fake_labels.cuda()
        bce_loss = nn.BCELoss()(D_fake, fake_labels)
        return self._l1_weight * l1_loss + self._bce_weight * bce_loss

    def sample_model(self, valid_data_gen):
        processed_images = 0
        for x_batch, y_batch in valid_data_gen:
            out_G = self.generator(x_batch)
            resImages = torch.cat([x_batch, y_batch, out_G], dim=2)
            oneImage = torch.cat([*resImages], dim=2)

            cur_batch_size = x_batch.shape[0]

            batch_image_name = os.path.join(self.IMAGE_LOGS, '%05d-%05d_batch.png' % (processed_images,
                                                                                      processed_images + cur_batch_size))
            DataGenerator.save_image(batch_image_name, oneImage.cpu())
            processed_images += cur_batch_size

    def train(self, n_epochs, batch_size, lr_G, lr_D, train_data_epoch_gen, valid_data_epoch_gen):
        if self.cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        g_loss_log = []
        d_loss_log = []
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

        best_G_loss = None
        for epoch in range(n_epochs):
            epoch_time = time()
            print('Epoch %d/%d\n' % (epoch, n_epochs))
            train_data_gen = train_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=self.cuda)

            logs = self.train_epoch(opt_G, opt_D, train_data_gen, len(train_data_epoch_gen),
                                    D_learning=epoch % 2 == 0)
            g_loss_log.extend(logs[0])
            d_loss_log.extend(logs[1])
            epoch_time = time() - epoch_time
            print('Train G-loss %f, D-loss %f for %fs' % (np.mean(logs[0]), np.mean(logs[1]), epoch_time))

            self.test_model(valid_data_epoch_gen.get_epoch_generator(batch_size=batch_size, cuda=self.cuda))

            if best_G_loss is None or np.mean(logs[0]) < best_G_loss:
                best_G_loss = np.mean(logs[0])
                torch.save(self.generator, self.BEST_GENERATOR)
                torch.save(self.discriminator, self.BEST_DISCRIMINATOR)
            if epoch % 10 == 0:
                generator_model_name = 'generator_epoch%05d.mdl' % epoch
                discriminator_model_name = 'discriminator_epoch%05d.mdl' % epoch
                torch.save(self.generator, os.path.join(self.MODEL_LOGS, generator_model_name))
                torch.save(self.discriminator, os.path.join(self.MODEL_LOGS, discriminator_model_name))
            self.sample_model(valid_data_epoch_gen.get_epoch_generator(batch_size=8, cuda=self.cuda))

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

    if activation is not None:
        layers += [activation]

    if use_bn and not bn_before:
        layers += [nn.BatchNorm2d(out_ch)]

    return layers


def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find(
            'BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class GeneratorP2P(nn.Module):
    def __init__(self, n_blocks=5, filters=32):
        super().__init__()
        cur_block = None
        cur_filters = 2**(n_blocks - 2) * filters

        for i in range(n_blocks - 2):
            cur_block = UNetBlock(cur_filters, cur_block)
            cur_filters //= 2
        cur_block = UNetBlock(in_ch=3, sub_block=cur_block, first_filters=cur_filters * 2, last_filters=cur_filters)
        self.model = nn.Sequential(
            cur_block,
            *conv_block(cur_filters + 3, cur_filters, 3, 1, stride=1, use_bn=True, bn_before=True),
            *conv_block(cur_filters, 3, 1, 0, stride=1, activation=nn.Tanh(), use_bn=False, bn_before=True)
        )
        self.model.apply(init_func)


    def forward(self, x):
        return self.model(x)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, sub_block=None, first_filters=None, last_filters=None):
        super().__init__()
        kernel_size = 3
        padding = kernel_size // 2
        layers = []

        down_ch = 2 * in_ch if first_filters is None else first_filters

        layers += conv_block(in_ch, down_ch, kernel_size, padding, stride=2, use_bn=False, bn_before=True)

        #print('Down in: %d\nDown out: %d' % (in_ch, down_ch))
        if sub_block is None:
            sub_block = conv_block(down_ch, 2 * down_ch, kernel_size, padding, stride=1, use_bn=False, bn_before=True)
        else:
            sub_block = [sub_block]
        layers += sub_block

        output_ch = in_ch if last_filters is None else last_filters
        layers += conv_block(2 * down_ch, output_ch, kernel_size + 1, padding=1, stride=2, transpose=True,
                             use_bn=True, bn_before=True)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = self.model(x)
        #print(out.shape)
        # connect over channels
        out_with_skip = torch.cat([x, out], dim=1)
        return out_with_skip

class DiscriminatorP2P(nn.Module):
    def __init__(self, conv_blocks_count=5):
        super().__init__()
        layers = []
        cur_inputs = 3 * 2
        cur_outputs = 0
        filters = 32
        kernel_size = 5
        padding = kernel_size // 2

        layers += conv_block(cur_inputs, filters, 7, 3, stride=1, use_bn=True, bn_before=True)
        cur_inputs = filters
        # stride convs
        for i in range(conv_blocks_count):
            cur_outputs = min(2**i, 8) * filters
            layers += conv_block(cur_inputs, cur_outputs, kernel_size, padding, stride=2, use_bn=False, bn_before=True)
            #layers += conv_block(cur_outputs, cur_outputs, kernel_size, padding, stride=1, use_bn=True, bn_before=True)
            cur_inputs = cur_outputs

        layers += conv_block(cur_outputs, 1, 1, 0, stride=1, activation=nn.Sigmoid(), use_bn=False, bn_before=True)


        self.model = nn.Sequential(*layers)

        self.model.apply(init_func)
        #self.layers = layers


    def forward(self, inp, target):
        x = torch.cat([inp, target], dim=1)
        return self.model(x)
