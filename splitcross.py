"""
Modified by Pan Yicheng  @ Oct, 16, 2019
For torch==1.3.0
"""
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np


class SplitCrossEntropyLoss(nn.Module):
    r"""SplitCrossEntropyLoss calculates an approximate softmax"""
    def __init__(self, hidden_size, splits, tied_weights=False, verbose=False):
        # We assume splits is [0, split1, split2, N] where N >= TokenNum
        # For example, a vocab of 1000 words may have splits [0] + [100, 500] + [inf]
        super(SplitCrossEntropyLoss, self).__init__()
        # hidden_size is the feature size of previous layer
        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        self.verbose = verbose
        # Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
        # The probability given to this tombstone is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            self.tail_vectors = nn.Parameter(torch.zeros(self.nsplits - 1, hidden_size))
            if not tied_weights:
                self.tail_bias = nn.Parameter(torch.zeros(self.nsplits - 1))
            else:
                self.tail_bias = None

    def logprob(self, weight, bias, hiddens, splits=None, softmaxed_head_res=None, verbose=False):
        # If not provided with softmaxed_head_res, we need to caculate it.
        if softmaxed_head_res is None:
            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if (end - start == 0 or (bias is None)) else bias[start:end]
            # We only add the tombstones if we have more than one split
            if self.nsplits > 1:
                head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
                head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

            # Perform the softmax calculation for  ```hiddens```
            head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
            softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)

        if splits is None:
            splits = list(range(self.nsplits))

        results = []
        running_offset = 0
        for idx in splits:

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])

            # If the target is in one of the splits, 
            # the probability is the p(tombstone) * p(word within tombstone)
            else:
                start, end = self.splits[idx], self.splits[idx + 1]
                tail_weight = weight[start:end]
                tail_bias = None if (bias is None) else bias[start:end]

                # Calculate the softmax for the words in the tombstone
                tail_res = torch.nn.functional.linear(hiddens, tail_weight, bias=tail_bias)
                tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1)
                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding in log space is equivalent to multiplication
                # The below extracts the P(tombstone) of the idx'th split
                head_entropy = (softmaxed_head_res[:, -(self.nsplits - idx)]).contiguous()
                results.append(head_entropy.view(-1, 1) + tail_entropy)

        if len(results) > 1:
            return torch.cat(results, dim=1)
        return results[0]

    def split_on_targets(self, hiddens, targets):
        """
        Split the targets into those in the head and in the tail
        :param hiddens: output of the rnn
        :param targets: array of token id of words
        :return: split_targets, split_hiddens

        """
        split_targets = []
        split_hiddens = []

        # Determine to which split each element belongs
        #  For example, targets = [0, 100, 200, 300], splits = [0, 100, 200, 300]
        # then                       mask = [0, 1,      2,      2]
        # Other method1:
        # This method appears slower at least for WT-103 values for approx softmax
        # masks = [(targets >= self.splits[idx]).view(1, -1) for idx in range(1, self.nsplits)]
        # mask = torch.sum(torch.cat(masks, dim=0), dim=0)
        #  Other method2:
        # masks = torch.stack([targets] * (self.nsplits - 1))
        # mask = torch.sum(masks >= self.split_starts, dim=0)
        mask = None
        for idx in range(1, self.nsplits):
            # for each split lower bound except  0, add 1 if equal or greater
            partial_mask = targets >= self.splits[idx]
            mask = mask + partial_mask.int() if mask is not None else partial_mask.int()
        # Split according to the mask
        processed_length = 0
        for idx in range(self.nsplits):
            # If there are no splits, avoid costly masked select
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                break
            # If all the words are covered by earlier splits,
            # so we append empty list that later stages don't freak out
            # improve efficiency by remove the sum
            if processed_length == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue
            # Are you in our split?
            tmp_mask = mask == idx
            split_targets.append(targets.masked_select(tmp_mask))
            split_hiddens.append(hiddens.masked_select(tmp_mask.unsqueeze(1).
                                                       expand_as(hiddens)).view(-1, self.hidden_size))
            processed_length += len(split_targets[-1])
        return split_targets, split_hiddens

    def forward(self, weight, bias, hiddens, targets, verbose=False):
        total_loss = None
        if len(hiddens.size()) > 2:
            hiddens = hiddens.view(-1, hiddens.size(2))

        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

        # First we perform the first softmax on the head vocabulary and the tombstones
        start, end = self.splits[0], self.splits[1]
        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if (end - start == 0 or (bias is None)) else bias[start:end]

        # We only add the tombstones if we have more than one split
        if self.nsplits > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

        # Perform the softmax calculation for the word vectors in the head for all splits
        # We need to guard against empty splits as torch.cat does not like random lists
        combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])
        all_head_res = torch.nn.functional.linear(combo, head_weight, bias=head_bias)
        softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)

        running_offset = 0
        for idx in range(self.nsplits):
            # If there are no targets for this split, continue
            if len(split_targets[idx]) == 0:
                continue

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
                # torch.gather select the target indexed element in softmaxed_head_res
                entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))
            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]

                # Calculate the softmax for the words and p(tombstone)*p(words in tombstone)
                tail_res = self.logprob(weight, bias, split_hiddens[idx], splits=[idx],
                                        softmaxed_head_res=softmaxed_head_res)

                # All indices are shifted - if the first split handles [0,...,499] 
                # then the 500th in the second split will be 0 indexed
                indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)
                entropy = -torch.gather(tail_res, dim=1, index=indices)
            running_offset += len(split_hiddens[idx])
            total_loss = entropy.float().sum() if total_loss is None else total_loss + entropy.float().sum()

        return total_loss / len(targets)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    V = 8
    H = 10
    N = 100
    E = 10

    embed = torch.nn.Embedding(V, H)
    crit = SplitCrossEntropyLoss(hidden_size=H, splits=[V // 2], verbose=True)
    bias = torch.nn.Parameter(torch.ones(V))
    optimizer = torch.optim.SGD(list(embed.parameters()) + list(crit.parameters()), lr=1)

    x = torch.autograd.Variable((torch.rand(N, 1) * 0.999 * V).int().long())
    prev = torch.autograd.Variable((torch.rand(N, 1) * 0.999 * V).int().long())
    print('X:', x)
    print('Previous:', prev)
    for _ in range(E):
        y = embed(prev).squeeze()
        c = crit(embed.weight, bias, y, x.view(N))
        print('Crit:', c.exp().data.item())

        probs = crit.logprob(embed.weight, bias, y[:2]).exp()
        print(probs)
        print(probs.sum(dim=1))

        optimizer.zero_grad()
        c.backward()
        optimizer.step()
