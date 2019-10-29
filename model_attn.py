import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.target_embedding = nn.Embedding(ntoken, emsize)
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
            self.decoder.weight = self.input_embedding.weight
        else:
            self.decoder = nn.Linear(self.nhid, self.ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        if not self.tie_weights:
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, targets, input_lens, target_lens,
                return_decoder_all_h=False,
                use_teacher_forcing=False, SOS_index=0):
        """
        input shape: (S, N)
        targets shape: (S, N)
        return_decoder_all_h: whether return every sequence value in decoder rnns
        """
        batch_size=input.size()[1]
        emb = embedded_dropout(self.input_embedding, input,
                               dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        # emb shape: (S, N, emsize)
        encoder_hidden = self.init_hidden(input.size()[1])
        packed_emb = pack_padded_sequence(emb, input_lens, batch_first=False,
                enforce_sorted=False)
        encoder_outputs, encoder_hidden = self.encoder_rnns(packed_emb, encoder_hidden)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs)
        encoder_outputs = self.lockdrop(encoder_outputs, self.dropout)
        # encoder_outputs shape: (S, N, nhid)
        # encoder_hidden shape: (nlayers*directions, N, nhid)
        decoder_rnns_output_list = []
        decoder_rnns_h_list = []
        decoder_input = self.input_embedding.weight.new_full([1, input.size()[1]],
                                                             SOS_index, dtype=torch.long)
        # decoder_input shape: (1, N)
        decoder_hidden = encoder_hidden
        for seq_index in range(input.size()[0]):
            decoder_input = self.target_embedding(decoder_input)
            h_n_batchfirst = decoder_hidden.transpose(0, 1)
            h_n_batchfirst = h_n_batchfirst.reshape(batch_size, -1)
            # h_n_batchfirst shape: (N, nlayers*directions*nhid)
            attn_weights = F.softmax(self.attn(
                torch.cat(
                    (decoder_input.view(-1, decoder_input.size()[2]),
                        h_n_batchfirst), dim=1))[:, :encoder_outputs.size()[0]])
            # attn_weights shape: (N, S)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                     encoder_outputs.transpose(0, 1))
            # attn_applied shape: N, 1, nhid
            attn_combine_output = F.relu(
                self.attn_combine(
                    torch.cat(
                        (decoder_input.view(-1, decoder_input.size()[2]),
                         attn_applied.view(
                             attn_applied.size()[0],
                             attn_applied.size()[2])),
                        dim=1)
                )
            )
            # attn_combine_output shape: N, nhid
            decoder_rnns_output, decoder_hidden = self.decoder_rnns(
                attn_combine_output.unsqueeze(0), decoder_hidden)
            # decoder_rnns_output shape: (1, N, nhid),
            # decoder_hidden shape: (nlayers*directions, N, nhid)
            decoder_rnns_output = self.decoder(decoder_rnns_output)
            # decoder_rnns_output shape: (1, N, ntok)
            if use_teacher_forcing:
                decoder_input = targets[seq_index].view(-1, batch_size)
            else:
                topv, topi = decoder_rnns_output.topk(1, dim=2)
                decoder_input = topi.view(1, batch_size).detach()

            decoder_rnns_output_list.append(decoder_rnns_output)
            decoder_rnns_h_list.append(decoder_hidden)
        decoder_rnns_output_tensor = torch.cat(tuple(decoder_rnns_output_list), dim=0)
        # decoder_rnns_output_tensor shape: (S, N, ntok)
        if not return_decoder_all_h:
            return decoder_rnns_output_tensor, decoder_hidden
        else:
            return decoder_rnns_output_tensor, decoder_hidden, \
                   decoder_rnns_h_list

    def init_hidden(self, bsz):
        weight = next(self.encoder_rnns.parameters()).data
        return weight.new(self.nlayers, bsz, self.nhid).zero_()
