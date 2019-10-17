"""
Apply dropout to a sequence but keep dropout mask the same
for every element in the sequence
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.2):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


if __name__ == '__main__':
    lockdropout = LockedDropout()
    dropout = nn.Dropout(p=0.5)

    x = torch.rand(3, 2, 5)

    x1 = dropout(x)
    print('torch.nn.Dropout:\n', x1)
    x2 = lockdropout(x, dropout=0.5)
    print('locked dropout:\n', x2)