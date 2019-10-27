import os
import torch
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.MAX_LENGTH = None
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.frequence_fix()

    def tokenize(self, path):
        """Tokenize a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # append <eos> to every sample sequence
                words = line.split() + ['<eos>']
                # Calculate max sequence length
                if self.MAX_LENGTH is None:
                    self.MAX_LENGTH = len(line[:line.find('<EOL>')+5].split(' '))
                else:
                    self.MAX_LENGTH = max(
                        len(line[:line.find('<EOL>') + 5].split(' ')),
                        self.MAX_LENGTH)
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                # append <eos> to every sample sequence
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def frequence_fix(self):
        #  Adjusting word id according to its frequencies
        ntokens = len(self.dictionary.idx2word)
        counter_list = []
        for i in range(ntokens):
            counter_list.append((i, self.dictionary.counter[i]))
        counter_list = sorted(counter_list, key=lambda x: x[1], reverse=True)
        map2newid = dict()
        for i, i_freq in enumerate(counter_list):
            map2newid[i_freq[0]] = i
        for i in range(self.train.size()[0]):
            self.train[i] = map2newid[self.train[i].item()]
        for i in range(self.valid.size()[0]):
            self.valid[i] = map2newid[self.valid[i].item()]
        for i in range(self.test.size()[0]):
            self.test[i] = map2newid[self.test[i].item()]
        old_idx2word = self.dictionary.idx2word
        new_idx2word = []
        for i in range(ntokens):
            new_idx2word.append(old_idx2word[counter_list[i][0]])
            self.dictionary.counter[i] = counter_list[i][1]
        self.dictionary.idx2word = new_idx2word
        for i in range(ntokens):
            self.dictionary.word2idx[new_idx2word[i]] = i

    def get_words(self, dataset, start_index, seq_len):
        if dataset == 'train':
            return [self.dictionary.idx2word[i]
                    for i in self.train[start_index:start_index+seq_len]]
        elif dataset == 'valid':
            return [self.dictionary.idx2word[i]
                    for i in self.valid[start_index:start_index+seq_len]]
        elif dataset == 'test':
            return [self.dictionary.idx2word[i]
                    for i in self.test[start_index:start_index+seq_len]]
        else:
            print('dataset name not error, must be one of '
                  'train, valid, test')
            return []

