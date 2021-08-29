# -*- coding: utf-8 -*-

from collections import namedtuple

import torch
import math

from torch import Tensor
from typing import List, Sequence

from torch.nn import Linear, Module, Embedding, Parameter, init
import torch.nn.functional as F

_ASMoutput = namedtuple('ASMoutput', ['output', 'loss'])

class SplitSoftmaxWithLoss(Module):
    r"""Efficient softmax approximation

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        tied_weights (torch.nn.Parameter): Embedding's weight to make use of tied weights
        bias (bool, optional): If ``True``, adds a bias term to the calculation. Default: ``True``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})`
        - target: :math:`(N)` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)`
        - output2: ``Scalar``
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        tied_weights: Parameter = None,
        bias: bool = True
    ) -> None:
        super(SplitSoftmaxWithLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs)
        self.weight = Parameter(torch.randn(n_classes, in_features))

        if bias:
            self.bias = Parameter(torch.zeros(n_classes))

        if self.n_clusters > 1:
            self.tail_vectors = Parameter(torch.zeros(self.n_clusters - 1, in_features))
            if bias:
                self.tail_bias = Parameter(torch.zeros(self.n_clusters - 1))
        
        if tied_weights is not None:
            self.weight = tied_weights
        else:
            self.reset_parameters(self.weight)

    def reset_parameters(self, param) -> None:
        init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(self, input: Tensor, target: Tensor, reduce_fn=torch.mean) -> _ASMoutput:
        
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)
        cutoff_values = [0] + self.cutoffs

        start, end = cutoff_values[0], cutoff_values[1]
        head_weight = self.weight[start:end]
        head_bias = self.bias[start:end] if hasattr(self, 'bias') else None

        if self.n_clusters > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = None if head_bias is None else torch.cat([head_bias, self.tail_bias])

        head_output = F.linear(input, head_weight, bias=head_bias)
        head_logprob = F.log_softmax(head_output, dim=1)

        for i in range(len(cutoff_values) - 1):
            
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = torch.nonzero(target_mask).squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                tail_weight = self.weight[low_idx:high_idx]
                tail_bias = self.bias[low_idx:high_idx] if hasattr(self, 'bias') else None
                cluster_output = F.linear(input_subset, tail_weight, bias=tail_bias)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = reduce_fn(-output)

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):

            tail_weight = self.weight[start_idx:stop_idx]
            tail_bias = self.bias[start_idx:stop_idx] if hasattr(self, 'bias') else None

            cluster_output = F.linear(input, tail_weight, bias=tail_bias)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r""" Computes log probabilities for all :math:`\texttt{n\_classes}`

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """

        start, end = 0, self.shortlist_size
        head_weight = self.weight[start:end]
        head_bias = self.bias[start:end] if hasattr(self, 'bias') else None

        if self.n_clusters > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = None if head_bias is None else torch.cat([head_bias, self.tail_bias])

        head_output = F.linear(input, head_weight, bias=head_bias)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r""" This is equivalent to `self.log_pob(input).argmax(dim=1)`,
        but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """

        start, end = 0, self.shortlist_size
        head_weight = self.weight[start:end]
        head_bias = self.bias[start:end] if hasattr(self, 'bias') else None

        if self.n_clusters > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = None if head_bias is None else torch.cat([head_bias, self.tail_bias])

        head_output = F.linear(input, head_weight, bias=head_bias)
        output = torch.argmax(head_output, dim=1)
        
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output


if __name__ == '__main__':
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    V = 4096 * 8
    H = 16
    N = 10
    E = 10

    embed = torch.nn.Embedding(V, H)

    # Cutoffs list calculation could be better than this
    start = 4 + int(np.ceil(np.log2(V / 8) / 2))
    cutoffs = [ 2**x for x in range(start, start + 5, 2) if V > 2**x ]
    crit = SplitSoftmaxWithLoss(H, V, cutoffs, embed.weight)

    print(V)
    print(crit.cutoffs)

    params = set(embed.parameters()).union(set(crit.parameters()))
    optimizer = torch.optim.SGD(params, lr=1)

    for _ in range(E):
        prev = torch.autograd.Variable((torch.rand(N, 1) * 0.999 * V).int().long())
        x = torch.autograd.Variable((torch.rand(N, 1) * 0.999 * V).int().long())
        y = embed(prev).squeeze()
        c = crit(y, x.view(N))

        print('Split Softmax', c.loss.item())
        with torch.no_grad():
            print('Cross Entropy', F.cross_entropy(F.linear(y, embed.weight, crit.bias), x.view(N)).item())

        print('Log prob')

        print(crit.log_prob(y)[torch.arange(y.size(0)), x.view(-1)])
        print('Cross Entropy log probs: ', F.log_softmax(F.linear(y, embed.weight, crit.bias), dim=1)[torch.arange(y.size(0)), x.view(-1)])

        optimizer.zero_grad()
        c.loss.backward()
        optimizer.step()