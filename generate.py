"""
This file generates new sentences sampled from the language model
"""
import argparse
import hashlib
import os
import torch
from torch.autograd import Variable
import data

parser = argparse.ArgumentParser(description='PyTorch  Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/rocstory_data',
                    help='location of the data corpus')
parser.add_argument('--conditional_data', type=str, default='',
                    help='location of the file that contains the content that '
                         'the generation conditions on')
parser.add_argument('--print_cond_data', action='store_true',
                    help='whether to print the prompt on which conditionally '
                         'generated text is conditioned')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--nsents', type=int, default=1000,
                    help='number of lines to generate')
parser.add_argument('--words', type=int, default=100,
                    help='number of maximum words for one test line')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, criterion, optimizer = torch.load(f)
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
eos_id = corpus.dictionary.word2idx('<eos>')
hidden = model.init_hidden(1)
input = torch.rand(1, 1).mul(ntokens).long()
if args.cuda:
    input = input.cuda()
# TODO: deal with words not  appeared in train data
cond_data = corpus.tokenize(args.conditional_data)
with open(args.outf, 'w') as outf:
    for nsent in range(args.nsents):
        try:
            # the only thing that breaks the while loop is if there are no more eos_ids
            idx = cond_data.index(eos_id)
        except:
            break
        cond_length = idx
        for i in range(args.words):
            if i < cond_length:
                word_idx = cond_data[i]
            input.data.fill_(word_idx)
            output, hidden = model(input, hidden)
            output = model.decode(output)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word = corpus.dictionary.idx2word[word_idx]
            if word_idx == eos_id:
                outf.write('\n')
                break
            outf.write(word)
        cond_data = cond_data[idx+1:]
        if nsent % args.log_interval == 0:
            print('Generated {}/{} words'.format(nsent, args.nsents))
