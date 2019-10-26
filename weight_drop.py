"""
Weight drop for hidden to hidden matrix in RNN models
TODO: need to reimplement in torch 1.3.0, bugs exist. Not used now.
1.Try rewrite the forward process of underlying RNN
2. See Keras LSTM source code implementation of it
"""
import torch
from torch.nn import Parameter
import warnings

warnings.filterwarnings("ignore")


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout: float = 0,
                 variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        # print('y2k')
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            # Delete original weight
            # delattr(self.module, name_w)
            # setattr(self.module, None)
            self.module.register_parameter(name_w + '_raw', Parameter(w))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            # raw_w = getattr(self.module, name_w)
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1, requires_grad=True)
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            w = torch.nn.Parameter(w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


if __name__ == '__main__':
    # Input is (seq, batch, input)
    x = torch.randn(8, 3, 10)
    x = x.cuda()
    target = torch.randint(0, 5, size=[x.size(0) * x.size(1)])
    print('target:\n', target)
    target = target.cuda()
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x)]
    run2 = [x.sum() for x in lin(x)]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    # wdrnn = torch.nn.LSTM(10, 10)
    wdrnn.cuda()
    wdrnn.module.flatten_parameters()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be
    # equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]
    print('---')

    print('Testing autograd of WeightDrop multilayer LSTM')
    wdrnn = WeightDrop(torch.nn.LSTM(10, 5, 2),
                       ['weight_hh_l{}'.format(i) for i in range(2)],
                       dropout=0.9)
    wdrnn.cuda()
    optimizer = torch.optim.SGD(wdrnn.parameters(), lr=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').cuda()

    for epoch in range(10):
        y, (h_n, c_n) = wdrnn(x)
        y = y.view(-1, y.size(2))
        loss = criterion(y, target)
        print('Loss: {:5.5f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wdrnn.eval()
    y, (h_n, c_n) = wdrnn(x)
    y = y.view(-1, y.size(2))
    loss = criterion(y, target)
    print('Loss after:{:5.5f}'.format(loss))
