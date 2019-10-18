import argparse
import time
import math
import numpy as np
import torch
import os
import hashlib

import data
import model
from utils import *
from splitcross import SplitCrossEntropyLoss

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./rocstory_data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=1,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
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


fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)
ntokens = len(corpus.dictionary)
print('Num tokens: {}'.format(ntokens))
eval_batch_size = 80
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
###############################################################################
# Build the model
###############################################################################
criterion = None

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                       args.dropoute, args.dropouti, args.dropoutrnn, args.dropout,
                       args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropoute, model.dropouti, model.dropout = args.dropoute, \
                                                                   args.dropouti, args.dropout
    # if args.whhdrop:
    #     from weight_drop import WeightDrop
    #     for rnn in model.rnns:
    #         if type(rnn) == WeightDrop:
    #             rnn.dropout = args.whhdrop
    #         elif rnn.zoneout > 0:
    #             rnn.zoneout = args.whhdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    elif ntokens > 20000:
        splits = [5000, 20000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.nhid, splits=splits,
                                      tied_weights=args.tied, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
print('{:-^60}'.format(''))
print('Args:', args)
print('{:-^60}'.format(''))
print('Model parameters:', count_parameters(model))
print('Criterion parameters:', count_parameters(criterion))
###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).item()
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN':
        model.reset()
    model.train()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden = model(data, hidden)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        # Activiation Regularization
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
            elapsed = time.time() - start_time
            cur_loss = total_loss / args.log_interval
            log_loss('loss_log.pkl', cur_loss, batch==args.log_interval)
            start_time = time.time()
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | {:5.2f} ms/batch  | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
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
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)!')
                stored_loss = val_loss
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
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
                print('Model all parameters:')
                optimizer = torch.optim.ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('{:-^60}'.format(''))
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
