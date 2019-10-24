# TODO: Use batch train, consider padding input and output sequence
from __future__ import unicode_literals, print_function, division
from io import open
import random
import hashlib
import argparse
import time
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description='PyTorch Language Model Attention')
parser.add_argument('--data', type=str, default='./rocstory_data/',
                    help='location of the data corpus')
parser.add_argument('--nencoderhid', type=int, default=400,
                    help='number of hidden units per layer in encoder')
parser.add_argument('--encoderlayers', type=int, default=3,
                    help='number of encoder layers')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        device = torch.device("cpu")
    else:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")


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
        self.train_pairs = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_pairs = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_pairs = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenize a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = 0
        for line in lines:
            # append <eos> to every sample sequence
            words = line.split() + ['<EOS>']
            input_words = line[:line.find('<EOL>')+5].split()
            output_words = line[line.find('<EOL>')+6:].split()
            # Calculate max sequence length
            if self.MAX_LENGTH is None:
                self.MAX_LENGTH = len(input_words)
            else:
                self.MAX_LENGTH = max(len(input_words), self.MAX_LENGTH)
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)
        self.dictionary.add_word('<SOS>')
    # Tokenize file content
        pairs = []
        token = 0
        for line in lines:
            input_words = line[:line.find('<EOL>') + 5].split()
            # append <EOS>  and prepend <SOS> to every output sequence
            output_words = ['<SOS>']+line[line.find('<EOL>') + 6:].split()+['<EOS>']
            pairs.append(([self.dictionary.word2idx[i] for i in input_words],
                          [self.dictionary.word2idx[i] for i in output_words]))
        return pairs

fn = 'corpus.{}.data'.format(hashlib.md5((args.data+'(lm_attention)').
                                         encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = Corpus(args.data)
    torch.save(corpus, fn)

SOS_token = corpus.dictionary.word2idx['<SOS>']
EOS_token =corpus.dictionary.word2idx['<EOS>']
MAX_LENGTH = corpus.MAX_LENGTH
num_tokens = len(corpus.dictionary)
print('Num of tokens       : {:>10}'.format(num_tokens))
print('Num of train pair : {:>10}'.format(len(corpus.train_pairs)))
print('Num of valid pair : {:>10}'.format(len(corpus.valid_pairs)))
print('Num of test pair   : {:>10}'.format(len(corpus.test_pairs)))


def tensorsFromPair(pair):
    input_tensor = torch.tensor(pair[0])
    target_tensor = torch.tensor(pair[1])
    return input_tensor, target_tensor


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_train_loss = float("inf")

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = corpus.train_pairs
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        for pair_index in range(len(training_pairs)):
            training_pair = tensorsFromPair(training_pairs[pair_index])
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('T: %s (Iter: %d %d%%) Loss: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            if print_loss_avg < best_train_loss:
                best_train_loss = print_loss_avg
                with open('lm_attention_model.pt', 'wb') as f:
                    torch.save([encoder, decoder], f)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt




def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        loss = 0
        criterion = torch.nn.NLLLoss(reduction='mean')
        for pair in corpus.valid_pairs:
            input_tensor = tensorsFromPair(pair)[0]
            target_tensor = tensorsFromPair(pair)[1]
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoder_outputs = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                decoder_outputs.append(decoder_output.squeeze())
                if topi.item() == EOS_token:
                    break
                decoder_input = topi.squeeze().detach()
            loss += criterion(torch.cat(decoder_outputs, dim=0), target_tensor).item()

        return loss / len(corpus.valid_pairs)


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(num_tokens, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, num_tokens,
                                   dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 500, print_every=10)

    val_loss = evaluate(encoder1, attn_decoder1)
    print('Val loss:{:5.3f}'.format(val_loss))
