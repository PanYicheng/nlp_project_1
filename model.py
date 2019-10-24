import torch
import torch.nn as nn
import torch.nn.functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nhid, nlayers,
                 dropoute=0.2, dropouti=0.2, dropoutrnn=0.2, dropout=0.2,
                 wdrop=0.5,
                 tie_weights=False,
                 use_attention=False, max_length=40):
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
        if wdrop:
            self.rnns = WeightDrop(self.rnns,
                                   ['weight_hh_l{}'.format(i) for i in range(nlayers)],
                                   wdrop)
        # attention layer
        if use_attention:
            self.attn = nn.Linear(nhid * (1 + nlayers), max_length)
            self.attn_combine = nn.Linear(nhid * 2, nhid)
            self.decoder_rnns = torch.nn.GRU(nhid, nhid, nlayers)
            if wdrop:
                self.decoder_rnns = WeightDrop(
                    self.decoder_rnns,
                    ['weight_hh_l{}'.format(i) for i in range(nlayers)], wdrop)

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
        self.use_attention = use_attention

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
        if self.use_attention:
            decoder_rnns_output_list = []
            for seq_index in range(emb.size()[0]):
                # output: S, N, nhid; h_n:nlayers, N, nhid
                h_n_batchfirst = h_n.transpose(0, 1)
                h_n_batchfirst = h_n_batchfirst.view(h_n_batchfirst.size()[0], -1)
                # h_n_batchfirst: N, (nlayers*nhid)
                attn_weights = F.softmax(self.attn(
                    torch.cat((emb[seq_index], h_n_batchfirst), dim=1)))
                attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                         output.transpose(0, 1))
                # attn_applied: N, 1, nhid
                attn_combine_output = F.relu(self.attn_combine(
                    torch.cat((emb[seq_index], attn_applied.squeeze()), dim=1)
                ))
                # attn_combine_output: N, nhid
                decoder_rnns_output, h_n = self.decoder_rnns(
                    attn_combine_output.unsqueeze(0), h_n)
                decoder_rnns_output_list.append(decoder_rnns_output)
            return torch.cat(tuple(decoder_rnns_output_list), dim=0), h_n
        else:
            output = output.view(output.size(0) * output.size(1), output.size(2))
            return output, h_n

    def init_hidden(self, bsz):
        weight = next(self.encoder.parameters()).data
        return weight.new(self.nlayers, bsz, self.nhid).zero_()
