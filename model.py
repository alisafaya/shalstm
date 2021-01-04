import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
from torch.nn import LayerNorm

# checkpoint = torch.utils.checkpoint.checkpoint
checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)

def attention(query, key, value, attn_mask=None, need_weights=True, dropout=None):
    
    batch_size, heads, query_len, dim = query.size()
    key_len = key.size(2)

    attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)

    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
        attention_scores = attention_scores + attn_mask

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout:
        attention_weights = dropout(attention_weights)

    attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)
    mix = torch.matmul(attention_weights, value)

    return mix, attention_weights


class Overparam(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.l1 = nn.Linear(nhid, 2 * nhid)
        self.nhid = nhid

    def forward(self, x):
        c, f = self.l1(x).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)

    
class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Boom(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None

        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.shortcut = shortcut
        self.act = GELU()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z


class Attention(nn.Module):
    def __init__(self, nhid, q=True, k=False, v=False, r=False, heads=1, dropout=None):
        super().__init__()
        self.qs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.ks = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.vs = nn.Parameter(torch.zeros(size=(1, 1, nhid), dtype=torch.float))
        self.qkvs = nn.Parameter(torch.zeros(size=(1, 3, nhid), dtype=torch.float))
        self.heads = heads
        self.nhid = nhid
        assert nhid % self.heads == 0, 'Heads must divide vector evenly'
        self.drop = nn.Dropout(dropout) if dropout else None
        self.gelu = GELU()
        self.q = nn.Linear(nhid, nhid) if q else None
        self.qln = LayerNorm(nhid, eps=1e-12)
        self.k = nn.Linear(nhid, nhid) if k else None
        self.v = nn.Linear(nhid, nhid) if v else None
        self.r = nn.Linear(2 * nhid, nhid) if r else None
        self.r_gate = nn.Parameter(torch.ones(size=(1, 1, nhid), dtype=torch.float))
        self.vq = None
        self.vq = Overparam(nhid)
        self.vq_collapsed = False

    def vq_collapse(self):
        vs = torch.sigmoid(self.vs)
        vs = self.vq(vs)
        self.vs.data = vs.data
        self.vq = None
        self.vq_collapsed = True

    def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
        qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
        if self.vq:
            vs = self.vq(vs)
        elif self.vq_collapsed:
            vs = self.vs

        if self.q:
            query = self.q(query)
            query = self.qln(query.float())
        if self.k:
            key = self.k(key)
        if self.v:
            value = self.v(value)
        
        q, k, v = qs * query, ks * key, vs * value

        if self.drop:
            q, k, v = self.drop(q), k, self.drop(v)

        original_q = q

        if not batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        batch_size, query_len, nhid = q.size()
        assert nhid == self.nhid
        key_len = k.size(1)
        ###
        dim = self.nhid // self.heads
        q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
        k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

        mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
        mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
        if not batch_first:
            mix = mix.transpose(0, 1)

        if self.r:
            r = torch.cat([mix, original_q], dim=-1)
            if self.drop:
                r = self.drop(r)
            
            r = self.gelu(self.r(r))
            mix = torch.sigmoid(self.r_gate) * mix + r

        return mix, focus


class Block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, heads=1, dropout=None, rnn=False, residual=True, use_attn=True):
        super().__init__()

        if use_attn:
            self.attn = Attention(embed_dim, heads=heads, r=False, dropout=dropout)
        else:
            self.attn = None
        
        self.ff = Boom(embed_dim, hidden_dim, dropout=dropout, shortcut=True)
        self.lnstart = LayerNorm(embed_dim, eps=1e-12)
        self.lnmid = LayerNorm(embed_dim, eps=1e-12)
        self.lnmem = LayerNorm(embed_dim, eps=1e-12)
        self.lnff = LayerNorm(embed_dim, eps=1e-12)
        self.lnxff = LayerNorm(embed_dim, eps=1e-12)
        self.drop = nn.Dropout(dropout)
        self.gelu = GELU()
        self.residual = residual
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=False)

    def forward(self, h, attn_mask, memory_size, mem=None, hidden=None):
        new_mem = None

        h = self.lnstart(h)

        if self.rnn:
            x, new_hidden = self.rnn(h, None if hidden is None else hidden) # checkpoint(self.rnn, h, None if hidden is None else hidden)
            ninp = h.shape[-1]
            z = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            z = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            x = self.drop(z).sum(dim=-2)

            h = h + x if self.residual else x.float()

        focus, new_mem = None, []

        if self.attn is not None:
            mh = self.lnmem(h)
            h = self.lnmid(h)

            if mem is not None:
                bigh = torch.cat([mem, mh], dim=0)
            else:
                bigh = mh
            new_mem = bigh[-memory_size:]

            q, k = h, bigh

            x, focus = checkpoint(self.attn, q, k, bigh, attn_mask)
            x = self.drop(x)
            h = x + h

        if self.ff:
            h, x = self.lnff(h), self.lnxff(h)
            x = checkpoint(self.ff, x)
            x = self.drop(x)
            h = x + h

        return h, new_mem, new_hidden, focus

    
class AdaptiveTiedEmbeddings(nn.AdaptiveLogSoftmaxWithLoss):
    def encode(self, input):
        
        used_rows = 0
        input_size = list(input.size())
        input = input.view(-1)
        device = self.head.weight.device
        output = torch.zeros(list(input.size()) + [self.in_features], device=device)
        cutoff_values = [0] + self.cutoffs

        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]
            input_mask = (input >= low_idx) & (input < high_idx)
            row_indices = input_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            cluster_input = input[input_mask] - low_idx

            if i == 0:
                out = torch.embedding(self.head.weight, cluster_input)
            elif self.div_value == 1:
                out = torch.embedding(self.tail[i - 1][1].weight, cluster_input)
            else:
#                 vector_idx = torch.tensor(self.shortlist_size + i - 1, dtype=torch.long, device=input.device)
                out = torch.embedding(self.tail[i - 1][1].weight, cluster_input)
                out = torch.matmul(out, self.tail[i - 1][0].weight) # * torch.embedding(self.head.weight, vector_idx)

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


class SHARNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, no_layers, memory_size=5120, dropout=0.5, dropouth=0.5, dropouti=0.5):
        super().__init__()

        self.embed_size = embed_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.no_layers = no_layers
        
        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        
#         self.encode = nn.Embedding(vocab_size, embed_size)
        
        if vocab_size > 256:
            start = int(np.ceil(np.log2(vocab_size / 32) / 2))
            cutoffs = [ 4**x for x in range(start, start + 5) if vocab_size > 2 * 4**x ]
        else:
            cutoffs = [32, 96]

        self.ate = AdaptiveTiedEmbeddings(embed_size, vocab_size, cutoffs)

        self.blocks = nn.ModuleList()
        for idx in range(no_layers):
            # place only one attention head on the layer before the last layer
            self.blocks.append(Block(embed_size, hidden_size, dropout=dropouth, use_attn=True if idx == no_layers - 2 else False))

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=1e1 / np.sqrt(self.hidden_size))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden=None, mems=None, targets=None):
        """ Input has shape [seq length, batch] """
        
        # encode and dropout input
        h = self.ate.encode(x)
#         h = self.encode(x)
        h = self.idrop(h)

        # if memory is provided, trim it to fit max memory size
        if mems is not None:
            maxmem = self.memory_size - len(h)
            mems = [m[-maxmem:] for m in mems]
        total_length = len(x) + (len(mems[0]) if mems else 0)

        # create attention mask
        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        if mems is not None:
            max_mems = max(len(m) for m in mems)
            happy = torch.zeros((len(x), max_mems), device=h.device, dtype=h.dtype)
            attn_mask = torch.cat([happy, attn_mask], dim=-1)

        # iterate over blocks 
        new_hidden, new_mems = [], []
        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems is not None else None
            hid = hidden[idx] if hidden is not None else None
            h, m, nh, f = block(h, attn_mask, self.memory_size, mem=mem, hidden=hid)
            new_hidden.append(nh)
            new_mems.append(m)
        
        # final dropout
        h = self.drop(h)

        if targets is not None:
            # calculate loss
            loss = self.ate(h.view(-1, self.embed_size), targets.view(-1)).loss
#             if torch.isnan(loss):
#                 import ipdb; ipdb.set_trace()
            return loss, h, new_hidden, new_mems
        else:
            # calculate predictions
            output = self.ate.predict(h.view(-1, self.embed_size))
            return output, h, new_hidden, new_mems
