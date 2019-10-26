import argparse
import time
import math
import numpy as np
import torch
import os
import hashlib
from collections import Counter

import model_attn
from utils import repackage_hidden, count_parameters, timeSince
import warnings
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

# default `log_dir` is "runs" - we'll be more specific here
tb_writer = SummaryWriter('runs/attention')
tb_writer_flag = False

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch RNN Attention-based'
                                             'Language Model')
parser.add_argument('--data', type=str, default='./rocstory_data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--evaluate_print_sample', action='store_true',
                    help='whether print one sample sentence when evaluating')
# TODO : remove the useless dropout
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutrnn', type=float, default=0.1,
                    help='amount of weight dropout to apply between RNN layers')
parser.add_argument('--tied', action='store_true',
                    help='tie projection matrix with embedding matrix')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='non monotonic history length to check')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0.001,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    if not os.path.exists(os.path.dirname(fn)):
        print('Making directory: {}'.format(os.path.dirname(fn)))
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            model, criterion, optimizer = torch.load(f)
    else:
        print('Error! Cannot load model from file {}'.format(fn))
        exit(1)





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
        self.IN_MAX_LENGTH = None
        self.OUT_MAX_LENGTH = None
        # <SOS> index, default to 0, same as model.forward()
        self.dictionary.add_word('<SOS>')
        # add ignore word, its id will be 1, the loss on target 1 can be ignored
        self.dictionary.add_word('<IGNORE>')
        self.train_pairs = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_pairs = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_pairs = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """
        Tokenize a text file.
        every pair in returned value will be like:
            t_0, t_1, ..., t_m, <EOT>, l_0, l_1, ..., l_n, <EOL>
            s_0, s_1, ..., s_p, <EOS>
        """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            lines = f.readlines()
        self.dictionary.add_word('<SOS>')
        for line in lines:
            # append <eos> to every sample sequence
            words = line.split() + ['<EOS>']
            input_words = line[:line.find('<EOL>')+5].split()
            output_words = line[line.find('<EOL>')+6:].split()+['<EOS>']
            # Calculate max sequence length
            if self.IN_MAX_LENGTH is None:
                self.IN_MAX_LENGTH = len(input_words)
            else:
                self.IN_MAX_LENGTH = max(len(input_words),
                                         self.IN_MAX_LENGTH)
            if self.OUT_MAX_LENGTH is None:
                self.OUT_MAX_LENGTH = len(output_words)
            else:
                self.OUT_MAX_LENGTH = max(len(output_words),
                                          self.OUT_MAX_LENGTH)
            for word in words:
                self.dictionary.add_word(word)
    # Tokenize file content
        pairs = []
        for line in lines:
            input_words = line[:line.find('<EOL>') + 5].split()
            # append <EOS>  and prepend <SOS> to every output sequence
            output_words = line[line.find('<EOL>') + 6:].split()+['<EOS>']
            pairs.append(([self.dictionary.word2idx[i] for i in input_words],
                          [self.dictionary.word2idx[i] for i in output_words]))
        return pairs

    def list_to_words(self, id_list):
        words = []
        for id in id_list:
            words.append(self.dictionary.idx2word[id])
        return words

    def list_to_sentence(self, id_list):
        for i, id in enumerate(id_list):
            if id == self.dictionary.word2idx['<IGNORE>']:
                break
        return ''.join(self.list_to_words(id_list[:i]))

    def pair_to_words(self, pair):
        input_words = self.list_to_words(pair[0])
        output_words = self.list_to_words(pair[1])
        return input_words, output_words

    @property
    def max_seq_length(self):
        return max(self.IN_MAX_LENGTH, self.OUT_MAX_LENGTH)


class MyDataset(torch.utils.data.Dataset):
    """
    convert words index list to torch tensor and
    padding to max sequence length
    """
    def __init__(self, pairs, max_seq_length, ignore_index):
        super(MyDataset, self).__init__()
        self.pairs = pairs
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index

    def __getitem__(self, key):
        pair = self.pairs[key]
        input_sequence = torch.full([self.max_seq_length], self.ignore_index,
                                    dtype=torch.long)
        output_sequence = torch.full([self.max_seq_length], self.ignore_index,
                                     dtype=torch.long)
        input_sequence[:len(pair[0])] = torch.tensor(pair[0], dtype=torch.long)
        output_sequence[:len(pair[1])] = torch.tensor(pair[1], dtype=torch.long)
        return input_sequence, output_sequence

    def __len__(self):
        return len(self.pairs)




fn = 'corpus.{}.data'.format(hashlib.md5((args.data+'(main_attn)').
                                         encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = Corpus(args.data)
    torch.save(corpus, fn)
ntokens = len(corpus.dictionary)
print('Num tokens: {}'.format(ntokens))
train_dataset = MyDataset(corpus.train_pairs, corpus.max_seq_length,
                          corpus.dictionary.word2idx['<IGNORE>'])
val_dataset = MyDataset(corpus.valid_pairs, corpus.max_seq_length,
                        corpus.dictionary.word2idx['<IGNORE>'])
test_dataset = MyDataset(corpus.test_pairs, corpus.max_seq_length,
                         corpus.dictionary.word2idx['<IGNORE>'])
train_data = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=args.batch_size, pin_memory=True)
val_data = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size, pin_memory=True)
test_data = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=1, pin_memory=True)
###############################################################################
# Build the model
###############################################################################
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=corpus.dictionary.word2idx['<IGNORE>'])

model = model_attn.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, args.dropoute, args.dropouti,
                       args.dropoutrnn, args.dropout, args.tied,
                       max_seq_length=corpus.max_seq_length)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropoute, model.dropouti, model.dropout = \
        args.dropoute, args.dropouti, args.dropout
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = model.parameters()
print('{:-^60}'.format(''))
print('Args:', args)
print('{:-^60}'.format(''))
print('Model parameters:', count_parameters(model))

###############################################################################
# Training code
###############################################################################

def evaluate(data_loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    for i, sample in enumerate(data_loader):
        inputs, targets = sample
        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        # inputs shape: (S, N)  targets shape: (S, N)
        outputs, hidden = model(inputs, targets, return_decoder_all_h=False,
                                use_teacher_forcing=False,
                                SOS_index=corpus.dictionary.word2idx['<SOS>'])
        # outputs shape: (S, N, ntok)
        if args.evaluate_print_sample and i==0:
            print('Inputs:\n---- ', corpus.list_to_sentence(inputs[:, 0]))
            outputs_word_ids = outputs.topk(1, dim=2)[:, 0, 0]
            print('Predicts:\n---- ', corpus.list_to_sentence(outputs_word_ids))
            print('Real words:\n---- ', corpus.list_to_sentence(targets[:, 0]))
        outputs = outputs.permute(1, 2, 0)
        targets = targets.permute(1, 0)
        total_loss += inputs.numel() * criterion(outputs, targets).item()
    return total_loss / len(data_loader.dataset)


def train(epoch):
    global tb_writer_flag
    model.train()
    total_loss = 0
    batches = int(len(train_data.dataset) / args.batch_size)
    batch_estimate_time = time.time()
    train_start_time = time.time()
    for batch, sample in enumerate(train_data):
        inputs, targets = sample
        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        if args.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        # write the first data to tensorboard
        if not tb_writer_flag:
            tb_writer.add_graph(model, (inputs, targets))
            tb_writer_flag = True
        optimizer.zero_grad()
        output, hidden = model(inputs, targets, return_decoder_all_h=False,
                               use_teacher_forcing=False,
                               SOS_index=corpus.dictionary.word2idx['<SOS>'])
        output = output.permute(1, 2, 0)
        targets = targets.permute(1, 0)
        #  output shape: (N, ntok, S), targets shape: (N, S)
        loss = criterion(output, targets)

        # TODO:Activiation Regularization
        # if args.alpha:
        #     loss = loss + args.alpha * output.pow(2).mean()
        # TODO: emporal Activation Regularization (slowness)
        # if args.beta:
        #     loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.data.item()
        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - batch_estimate_time
            batch_estimate_time = time.time()
            cur_loss = total_loss / args.log_interval
            tb_writer.add_scalar('training loss',
                                 cur_loss,
                                 batch+epoch*batches)
            print('| {:5d}/{:5d} batches | lr {:05.5f} '
                  '| {:5.2f} ms/batch  | loss {:5.2f} | ppl {:8.2f} '
                  '| bpc {:8.3f} | Time: {}'.
                  format(batch, batches, optimizer.param_groups[0]['lr'],
                         elapsed * 1000 / args.log_interval,
                         cur_loss, np.exp(cur_loss), cur_loss / np.log(2),
                         timeSince(train_start_time, batch / batches)
                         )
                  )
            total_loss = 0
        ###


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        if 't0' in optimizer.param_groups[0]:
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, np.exp(val_loss), val_loss / np.log(2)))
            print('-' * 89)
            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)!')
                stored_loss = val_loss
        else:
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, np.exp(val_loss), val_loss / np.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' \
                    and ('t0' not in optimizer.param_groups[0]) \
                    and (len(best_val_loss) > args.nonmono) \
                    and (val_loss > min(best_val_loss[-args.nonmono:])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save('{}.e{}'.format(args.save, epoch))
            print('Dividing learning rate by 10')
            optimizer.param_groups[0]['lr'] /= 10.
        tb_writer.add_scalar('valid_loss', val_loss, epoch)
        best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('{:-^60}'.format(''))
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
