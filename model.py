import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nhid, nlayers,
                 dropoute=0.2, dropouti=0.2, dropoutrnn=0.2, dropout=0.2,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, emsize)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = torch.nn.LSTM(emsize, nhid, nlayers, dropout=dropoutrnn)
        if rnn_type == 'GRU':
            self.rnns = torch.nn.GRU(emsize, nhid, nlayers, dropout=dropoutrnn)
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=emsize if l == 0 else nhid,
                                   hidden_size=nhid if l != nlayers - 1 else (emsize if tie_weights else nhid),
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in
                         range(nlayers)]
        print(self.rnns)


        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != emsize:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder = nn.Linear(nhid, ntoken, bias=False)
            self.decoder.weight = self.encoder.weight
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        self.ninp = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropout = dropout
        self.tie_weights = tie_weights

        self.init_weights()

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.tie_weights:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        output, h_n = self.rnns(emb, hidden)

        output = self.lockdrop(output, self.dropout)

        output = output.view(output.size(0) * output.size(1), output.size(2))
        return output, h_n

    def init_hidden(self, bsz):
        weight = next(self.encoder.parameters()).data
        return weight.new(self.nlayers, bsz, self.nhid).zero_()
