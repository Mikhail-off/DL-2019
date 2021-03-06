import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time

class ModelTrainer:
    def __init__(self, train_generator, test_generator):
        self.__model = None
        self.__train_generator = train_generator
        self.__test_generator = test_generator

    def set_model(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def train_epoch(self, optimizer, batch_size=32, cuda=True):
        assert self.__model is not None
        
        model = self.__model

        loss_log, acc_log = [], []
        model.train()
        steps = 0
        for batch_num, (x_batch, y_batch) in enumerate(self.__train_generator.get_epoch_generator(batch_size=batch_size)):
            data = x_batch.cuda() if cuda else x_batch
            target = y_batch.cuda() if cuda else y_batch

            optimizer.zero_grad()
            output = model(data)
            pred = torch.max(output, 1)[1].cpu()
            acc = torch.eq(pred, y_batch).float().mean()
            acc_log.append(acc)
            
            loss = F.nll_loss(output, target).cpu()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            loss_log.append(loss)
            
            steps += 1
            print('Step {0}'.format(steps), flush=True, end='\r')

        return loss_log, acc_log, steps
        

    def train(self, n_epochs, batch_size=32, lr=1e-3, cuda=True, plot_history=None, clear_output=None):
        assert self.__model is not None
    
        if cuda:
            self.__model = self.__model.cuda()
        else:
            self.__model = self.__model.cpu()

        model = self.__model
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        train_log, train_acc_log = [], []
        val_log, val_acc_log = [], []

        best_val_score = 0.

        for epoch in range(n_epochs):
            epoch_begin = time()
            print("Epoch {0} of {1}".format(epoch, n_epochs))
            train_loss, train_acc, steps = self.train_epoch(opt, batch_size=batch_size, cuda=cuda)

            val_loss, val_acc = self.test(cuda=cuda)

            train_log.extend(train_loss)
            train_acc_log.extend(train_acc)

            val_log.append((steps * (epoch + 1), np.mean(val_loss)))
            val_acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

            if np.mean(val_acc) > best_val_score:
                best_val_score = np.mean(val_acc)
                torch.save(model, 'models/model_best.mdl')
            
            if plot_history is not None:
                clear_output()
                plot_history(train_log, val_log)
                plot_history(train_acc_log, val_acc_log, title='accuracy')   
            epoch_end = time()
            epoch_time = epoch_end - epoch_begin
            print("Epoch: {2}, val loss: {0}, val accuracy: {1}".format(np.mean(val_loss), np.mean(val_acc), epoch))
            print("Epoch: {2}, train loss: {0}, train accuracy: {1}".format(np.mean(train_loss), np.mean(train_acc), epoch))
            print('Epoch time: {0}'.format(epoch_time))
        self.__model = model.cpu()

    def test(self, cuda=True):
        assert self.__model is not None
        
        model = self.__model
        
        loss_log, acc_log = [], []
        model.eval()
        
        for batch_num, (x_batch, y_batch) in enumerate(self.__test_generator.get_epoch_generator()):    
            data = x_batch.cuda() if cuda else x_batch
            target = y_batch.cuda() if cuda else y_batch

            output = model(data)
            loss = F.nll_loss(output, target).cpu()

            pred = torch.max(output, 1)[1].cpu()
            acc = torch.eq(pred, y_batch).float().mean()
            acc_log.append(acc)
            
            loss = loss.item()
            loss_log.append(loss)

        return loss_log, acc_log



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)