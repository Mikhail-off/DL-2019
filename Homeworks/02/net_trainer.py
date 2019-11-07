import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
import torch.nn.init as init
from tqdm import tqdm

from data_generator import start_idx, end_idx

SEQ_LIMIT = 20


class ModelTrainer:
    def __init__(self, train_generator, test_generator):
        self.__model = None
        self.__train_generator = train_generator
        self.__test_generator = test_generator

    def set_model(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def train_epoch(self, optimizer, is_force=False, cuda=True):
        assert self.__model is not None

        model = self.__model

        loss_log, acc_log = [], []
        model.train()
        steps = 0
        for batch_num, (x_batch, y_batch) in tqdm(enumerate(
                self.__train_generator.get_epoch_generator()), total=self.__train_generator.generator_steps):
            data = x_batch.cuda() if cuda else x_batch
            target = y_batch.cuda() if cuda else y_batch

            target_true = target[:, :-1]
            target_input = target[:, 1:]

            optimizer.zero_grad()
            output = model.forward_train(data, target_input, is_force=is_force)

            pred = torch.max(output, 1)[1]
            acc = torch.eq(pred, target_true).cpu().float().mean()
            acc_log.append(acc)

            loss = F.cross_entropy(output, target_true).cpu()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            loss_log.append(loss)

            steps += 1
            # print('Step {0}/{1}'.format(steps, self.__train_generator.generator_steps), flush=True, end='\r')

        return loss_log, acc_log, steps

    def train(self, n_epochs, lr=1e-3, is_force=False, cuda=True, plot_history=None, clear_output=None):
        assert self.__model is not None

        if cuda:
            self.__model = self.__model.cuda()
        else:
            self.__model = self.__model.cpu()

        model = self.__model
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        train_log, train_acc_log = [], []
        val_log, val_acc_log = [], []

        best_val_score = 0.

        for epoch in range(n_epochs):
            epoch_begin = time()
            print("Epoch {0} of {1}".format(epoch, n_epochs))
            train_loss, train_acc, steps = self.train_epoch(opt, cuda=cuda, is_force=is_force)

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
            print("Epoch: {2}, train loss: {0}, train accuracy: {1}".format(np.mean(train_loss), np.mean(train_acc),
                                                                            epoch))
            print('Epoch time: {0}'.format(epoch_time))
        self.__model = model.cpu()

    def test(self, cuda=True):
        assert self.__model is not None

        model = self.__model.cuda() if cuda else self.__model

        loss_log, acc_log = [], []
        model.eval()

        for batch_num, (x_batch, y_batch) in tqdm(enumerate(self.__test_generator.get_epoch_generator()),
                                                  total=self.__test_generator.generator_steps):
            data = x_batch.cuda() if cuda else x_batch
            target = y_batch.cuda() if cuda else y_batch

            target_true = target[:, :-1]

            output = model.forward_test(data)
            loss = F.cross_entropy(output, target_true).cpu()

            pred = torch.max(output, 1)[1]
            acc = torch.eq(pred, target_true).cpu().float().mean()
            acc_log.append(acc)

            loss = loss.item()
            loss_log.append(loss)

        return loss_log, acc_log


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        assert hidden_size % 2 == 0
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, word_inputs, hidden):
        embedded = self.embedding(word_inputs)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batches):
        return torch.zeros(2, 2 * self.n_layers, batches, int(self.hidden_size / 2))


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        init.normal_(self.embedding.weight, 0.0, 0.2)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)

    def forward(self, word_inputs, hidden):
        embedded = torch.unsqueeze(self.embedding(word_inputs), 1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden


class TranslationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, n_layers, is_cuda=True):
        super().__init__()

        self.is_cuda = is_cuda

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_vocab_size, hidden_size, self.n_layers)
        self.decoder = Decoder(output_vocab_size, hidden_size, self.n_layers)

        self.clf = nn.Linear(hidden_size, output_vocab_size)
        init.normal_(self.clf.weight, 0.0, 0.2)
        self.softmax = nn.Softmax(dim=-1)

    def _forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        if self.is_cuda:
            init_hidden = init_hidden.cuda()
        encoder_outputs, encoder_hidden = self.encoder(x, tuple(init_hidden))
        encoder_hidden_h, encoder_hidden_c = encoder_hidden

        self.decoder_hidden_h = encoder_hidden_h.permute(1, 0, 2).reshape(batch_size, self.n_layers,
                                                                          self.hidden_size).permute(1, 0, 2)
        self.decoder_hidden_c = encoder_hidden_c.permute(1, 0, 2).reshape(batch_size, self.n_layers,
                                                                          self.hidden_size).permute(1, 0, 2)
        return self.decoder_hidden_h.contiguous(), self.decoder_hidden_c.contiguous()

    def _forward_decoder_train(self, x, y, hidden_h, hidden_c, is_force=False):
        H = []

        current_y = torch.tensor([start_idx] * x.shape[0], dtype=torch.long)
        if self.is_cuda:
            current_y = current_y.cuda()
        for i in range(x.shape[1] - 1):
            inp = y[:, i] if is_force else current_y
            decoder_output, decoder_hidden = self.decoder(current_y, (hidden_h, hidden_c))
            hidden_h, hidden_c = decoder_hidden
            h = self.clf(decoder_output.squeeze(1))
            y_pred = self.softmax(h)
            _, current_y = torch.max(y_pred, dim=-1)
            H.append(h.unsqueeze(2))

        return torch.cat(H, dim=2)

    def _forward_decoder_test(self, x, hidden_h, hidden_c):
        return self._forward_decoder_train(x, None, hidden_h, hidden_c)

    def _forward_decoder(self, x, hidden_h, hidden_c):
        current_y = start_idx
        result = []
        limit = 0
        while current_y != end_idx and limit < SEQ_LIMIT:
            inp = torch.tensor([current_y])
            if self.is_cuda:
                inp = inp.cuda()
            decoder_output, decoder_hidden = self.decoder(inp, (hidden_h, hidden_c))
            hidden_h, hidden_c = decoder_hidden
            h = self.clf(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=-1)
            current_y = current_y.item()
            result.append(y)
            limit += 1
        result = torch.stack(result).unsqueeze(0)
        return result

    def forward_train(self, x, y, is_force=False):
        hidden_h, hidden_c = self._forward_encoder(x)
        return self._forward_decoder_train(x, y, hidden_h, hidden_c, is_force=is_force)

    def forward_test(self, x):
        hidden_h, hidden_c = self._forward_encoder(x)
        return self._forward_decoder_test(x, hidden_h, hidden_c)

    def forward(self, x):
        hidden_h, hidden_c = self._forward_encoder(x)
        return self._forward_decoder(x, hidden_h, hidden_c)
