from collections import Counter
import torch
import numpy as np

MAX_SENTENCES = 32

padding_token = '<pad>'
start_token = '<start>'
end_token = '<end>'
unknown_token = '<unk>'
padding_idx = 0
start_idx = 1
end_idx = 2
unknown_idx = 3

standart_tokens = [padding_token, start_token, end_token, unknown_token]

puncts = [',','.', '!', '?', ':']
min_word_count = 2

class VocabularyProcessor:
    def __init__(self, file_name):
        print('Opened', file_name)
        self.sentences = list()
        self.word2index = dict()
        self.index2word = dict()
        for i, word in enumerate(standart_tokens):
            self.word2index[word] = i
            self.index2word[i] = word
        assert self.word2index[padding_token] == padding_idx
        assert self.word2index[start_token] == start_idx
        assert self.word2index[end_token] == end_idx
        assert self.word2index[unknown_token] == unknown_idx

        self._initialize(file_name)

    def _initialize(self, file_name):
        counter = Counter()
        with open(file_name, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                sent = line.split()
                self.sentences.append(sent)
                for word in sent:
                    cur_len = len(self.word2index)
                    if word not in self.word2index:
                        self.word2index[word] = cur_len
                        self.index2word[cur_len] = word
                    counter[word] += 1
                line_num += 1
                if line_num == MAX_SENTENCES:
                    break

        assert len(self.word2index) == len(self.index2word)
        old_len = len(self.word2index)
        for word, index in list(self.word2index.items()):
            if counter[word] < min_word_count and word not in standart_tokens:
                self.index2word[self.word2index[word]] = None
                self.word2index[word] = None
        assert len(self.word2index) == len(self.index2word)
        print('Deleted %d words' % (old_len - len(self.word2index)))
        print('Current vocab len: %d' % len(self.word2index))
        self.sentences = np.array(self.sentences)

    def sentence2vector(self, sentence, seq_size):
        assert self.word2index[padding_token] == 0 and self.index2word[0] == padding_token
        vector = torch.zeros(seq_size)
        vector[0] = self.word2index[start_token]
        for i, word in enumerate(sentence):
            if i + 1 == seq_size - 1:
                vector[i + 1] = self.word2index[end_token]
                break
            if self.word2index.get(word) is not None:
                vector[i + 1] = self.word2index[word]

        return vector

    def vector2sentence(self, vector):
        sentance = []
        for ind in vector:
            index = int(ind)
            if index in self.index2word:
                sentance.append(self.index2word[index])
            else:
                sentance.append(unknown_token)
        return sentance


class DataGenerator:
    def __init__(self, data_file, target_file, batch_size, seq_len):
        self.data = VocabularyProcessor(data_file)
        self.target = VocabularyProcessor(target_file)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self.generator_steps = len(self.data.sentences) // self._batch_size
        assert len(self.data.sentences) == len(self.target.sentences)

    def get_epoch_generator(self):
        batch_size = self._batch_size
        seq_len = self._seq_len
        for i in range(0, len(self.data.sentences), batch_size):
            if i + batch_size >= len(self.data.sentences):
                return

            x_data = torch.zeros(batch_size, seq_len, dtype=torch.long)
            y_data = torch.zeros(batch_size, seq_len, dtype=torch.long)

            for ind, (x_sent, y_sent) in enumerate(zip(self.data.sentences[i: i + batch_size],
                                                       self.target.sentences[i: i + batch_size])):
                x_data[ind] = self.data.sentence2vector(x_sent, seq_len)
                y_data[ind] = self.target.sentence2vector(y_sent, seq_len)
            print(x_data, end='\n\n')
            yield x_data, y_data
