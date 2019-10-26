import torch
import torch.nn as nn
import torch.nn.functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, and an attention based decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nhid, nlayers,
                 dropoute=0.2, dropouti=0.2, dropoutrnn=0, dropout=0.2,
                 tie_weights=False, max_seq_length=40):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropout = dropout
        self.dropoutrnn = dropoutrnn
        self.tie_weights = tie_weights
        self.max_seq_length = max_seq_length



        self.lockdrop = LockedDropout()
        self.input_embedding = nn.Embedding(ntoken, emsize)
        assert rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.encoder_rnns = torch.nn.LSTM(emsize, nhid, nlayers, dropout=self.dropoutrnn)
            self.decoder_rnns = torch.nn.LSTM(nhid, nhid, nlayers, dropout=self.dropoutrnn)
        elif rnn_type == 'GRU':
            self.encoder_rnns = torch.nn.GRU(emsize, nhid, nlayers, dropout=self.dropoutrnn)
            self.decoder_rnns = torch.nn.GRU(nhid, nhid, nlayers, dropout=self.dropoutrnn)
        # attention layer
        self.attn = nn.Linear(self.nhid * (1 + self.nlayers), self.max_seq_length)
        self.attn_combine = nn.Linear(self.nhid * 2, self.nhid)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.tie_weights:
            if self.nhid != self.emsize:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder = nn.Linear(self.nhid, self.ntoken, bias=False)
            self.decoder.weight = self.encoinput_embeddingder.weight
        else:
            self.decoder = nn.Linear(self.nhid, self.ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        if not self.tie_weights:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_decoder_all_h=False):
        """
        input shape: (S, N)
        hidden shape: (nlayers*directions, N, nhid)
        return_decoder_all_h: whether return every sequence value in decoder rnns
        """
        emb = embedded_dropout(self.input_embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        # emb shape: (S, N, emsize)
        output, h_n = self.encoder_rnns(emb, hidden)
        output = self.lockdrop(output, self.dropout)
        # output shape: (S, N, nhid), h_n shape: (nlayers*directions, N, nhid)
        decoder_rnns_output_list = []
        decoder_rnns_h_list = []
        for seq_index in range(emb.size()[0]):
            h_n_batchfirst = h_n.transpose(0, 1)
            h_n_batchfirst = h_n_batchfirst.reshape(h_n_batchfirst.size()[0], -1)
            # h_n_batchfirst shape: (N, nlayers*directions*nhid)
            attn_weights = F.softmax(self.attn(
                torch.cat((emb[seq_index], h_n_batchfirst), dim=1)))
            # attn_weights shape: (N, S)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                        output.transpose(0, 1))
            # attn_applied shape: N, 1, nhid
            attn_combine_output = F.relu(
                self.attn_combine(
                    torch.cat((emb[seq_index], attn_applied.squeeze().view(
                        attn_applied.size()[0],
                        attn_applied.size()[2])),
                              dim=1)))
            # attn_combine_output shape: N, nhid
            decoder_rnns_output, h_n = self.decoder_rnns(
                attn_combine_output.unsqueeze(0), h_n)
            # decoder_rnns_output shape: (1, N, nhid), h_n shape: (nlayers*directions, N, nhid)
            decoder_rnns_output_list.append(decoder_rnns_output)
            decoder_rnns_h_list.append(h_n)
        
        if not return_decoder_all_h:
            return torch.cat(tuple(decoder_rnns_output_list), dim=0), h_n
        else:
            return torch.cat(tuple(decoder_rnns_output_list), dim=0), \
                    h_n, decoder_rnns_h_list

    def init_hidden(self, bsz):
        weight = next(self.encoder_rnns.parameters()).data
        return weight.new(self.nlayers, bsz, self.nhid).zero_()
