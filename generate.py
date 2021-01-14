# coding: utf-8

import argparse
import functools
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import hashlib

import data
from shalstm.model import SHALSTM
from shalstm.optim import MinTrustLamb

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save(model, f)

def model_load(fn):
    global model, optimizer
    with open(fn, 'rb') as f:
        m = torch.load(f)
        d = m.state_dict()
        model.load_state_dict(d, strict=False)

def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

fn = 'corpus.{}.data'.format(hashlib.md5("data/enwik8/".encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

batch_size = 1
eval_batch_size = 1
val_data = batchify(corpus.valid, eval_batch_size, 232)
ntokens = len(corpus.dictionary)
inpt = val_data[:4096, :].cuda()

model = SHALSTM(ntokens, 1024, 2048)
model_load("bin/small_model.pt")
model = model.cuda()
model.eval()

hidden = None
mems = None
output, h,  hidden, mems = model(inpt[:-1], hidden=hidden, mems=mems)
inp = inpt[-1:, :]

sequence = "".join(list(map(lambda x: chr(int(corpus.dictionary.idx2word[x])) if x != 20 else "\n", inpt.flatten().cpu() )))
sequence += "|-start-of-generated-text-|"

with torch.no_grad():
    for i in range(4096):
        output, h, hidden, mems = model(inp, hidden=hidden, mems=mems)
        output = top_k_top_p_filtering(output.view(-1), top_p=0.98).view(*output.shape)
        token_weights = F.softmax(output, dim=-1).squeeze()
        output_idx = torch.multinomial(token_weights, num_samples=1)[0]
        inp.data.fill_(output_idx)
        token = chr(int(corpus.dictionary.idx2word[output_idx])) if output_idx != 20 else "\n"
        sequence += token

print(sequence)
