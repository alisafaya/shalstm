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

        # assert not output.isnan().any()

        return output


if __name__ == '__main__':

    device = torch.device("cuda:0")
    adaptive = AdaptiveTiedEmbeddings(128, 512, [32, 128, 256], device=device, div_value=2)
    print("No of parameters =", sum(x.numel() for x in adaptive.parameters()))


# [W python_anomaly_mode.cpp:104] Warning: Error detected in LogSoftmaxBackward. Traceback of forward call that caused the error:
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/runpy.py", line 193, in _run_module_as_main
#     "__main__", mod_spec)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/runpy.py", line 85, in _run_code
#     exec(code, run_globals)
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 181, in <module>
#     spmd_main(env_dict, args.local_world_size, args.local_rank, args)
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 139, in spmd_main
#     run_proc(local_world_size, local_rank, args)
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 96, in run_proc
#     device=device
#   File "/scratch/users/asafaya19/shalstm/train/train.py", line 125, in train
#     loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
#     result = self.forward(*input, **kwargs)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/parallel/distributed.py", line619, in forward
#     output = self.module(*inputs[0], **kwargs[0])
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
#     result = self.forward(*input, **kwargs)
#   File "/scratch/users/asafaya19/shalstm/shalstm/model.py", line 127, in forward
#     loss = self.ate(h.view(-1, self.embed_size), targets.to(self.device).view(-1)).loss
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
#     result = self.forward(*input, **kwargs)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/modules/adaptive.py", line 214, in forward
#     head_logprob = log_softmax(head_output, dim=1)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/nn/functional.py", line 1605, in log_softmax
#     ret = input.log_softmax(dim)
#  (function _print_stack)
# Traceback (most recent call last):
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 139, in spmd_main
#     run_proc(local_world_size, local_rank, args)
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 96, in run_proc
#     device=device
#   File "/scratch/users/asafaya19/shalstm/train/train.py", line 128, in train
#     scaler.scale(loss).backward()
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/tensor.py", line 221, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph)
#   File "/kuacc/users/asafaya19/anaconda3/envs/ml-graphs/lib/python3.7/site-packages/torch/autograd/__init__.py", line 132, in backward
#     allow_unreachable=True)  # allow_unreachable flag
# RuntimeError: Function 'LogSoftmaxBackward' returned nan values in its 0th output.
# Traceback (most recent call last):
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 139, in spmd_main
#     run_proc(local_world_size, local_rank, args)
#   File "/scratch/users/asafaya19/shalstm/train/dist.py", line 96, in run_proc
#     device=device
#   File "/scratch/users/asafaya19/shalstm/train/train.py", line 190, in train
#     global_step += 1