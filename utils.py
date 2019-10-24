import torch
import os
import time
import math
import datetime


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_loss(filename, loss_value, first=True):
    dirname = os.path.dirname(filename)
    if len(dirname) != 0 and not os.path.exists(dirname):
        os.makedirs(dirname)
    if first:
        torch.save([float(loss_value)], filename)
    else:
        loss_list = torch.load(filename)
        loss_list.append(float(loss_value))
        torch.save(loss_list, filename)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (datetime.timedelta(seconds=s),
                          datetime.timedelta(seconds=rs))


if __name__ == '__main__':
    print('{:-^60}'.format('log_loss test'))
    for i in range(100):
        log_loss('test.pkl', float(i), i == 0)
    loss_list = torch.load('test.pkl')
    print(loss_list)
    os.remove('test.pkl')

    start = time.time()
    time.sleep(1)
    print(timeSince(start, 0.1))
