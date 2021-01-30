import torch
import torch.nn as nn

class AdaptiveTiedEmbeddings(nn.AdaptiveLogSoftmaxWithLoss):
    """
    AdaptiveTiedEmbeddings

    This is used to utilize weights of adaptive softmax as embedding vectors, 
    using this we can use tied weights as adaptive input representations.

    See: https://pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    """

    def __init__(self, *args, device=torch.device("cpu"), **kwargs):
        super(AdaptiveTiedEmbeddings, self).__init__(*args, **kwargs)
        self.device = device
        self.to(device)

    def encode(self, input):

        input = input.to(self.device)
        input_size = list(input.size())
        used_rows = 0

        input = input.view(-1)
        output = torch.zeros(list(input.size()) + [self.in_features], device=self.device)
        cutoff_values = [0] + self.cutoffs

        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]
            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = torch.nonzero(input_mask).squeeze()

            if row_indices.numel() == 0:
                continue

            cluster_input = input[input_mask] - low_idx
            if i == 0:
                out = torch.embedding(self.head.weight, cluster_input)
            elif self.div_value == 1:
                out = torch.embedding(self.tail[i - 1][1].weight, cluster_input)
            else:
                vector_idx = torch.tensor(self.shortlist_size + i - 1, dtype=torch.long, device=self.device)
                out = torch.embedding(self.tail[i - 1][1].weight, cluster_input)
                out = torch.matmul(out, self.tail[i - 1][0].weight) * torch.embedding(self.head.weight, vector_idx)

            output[row_indices] += out.squeeze()
            used_rows += row_indices.numel()

        if used_rows != input.size()[0]:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     input.min().item(),
                                                     input.max().item()))

        output = output.view(input_size + [self.in_features])
        return output


if __name__ == '__main__':

    device = torch.device("cuda:0")
    adaptive = AdaptiveTiedEmbeddings(128, 512, [32, 128, 256], device=device, div_value=2)
    print("No of parameters =", sum(x.numel() for x in adaptive.parameters()))