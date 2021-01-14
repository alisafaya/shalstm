import math 

import torch
import torch.nn as nn
import torch.nn.functional as F


class Overparam(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        c, f = self.linear(x).split(self.hidden_size, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size

        # gates
        self.query_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.key_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.value_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))

        # over parameterized values gate
        self.overparameterize = Overparam(hidden_size)

        # only applies to query
        self.affine_query = nn.Linear(hidden_size, hidden_size)
        self.ln_query = nn.LayerNorm(hidden_size, eps=1e-12)

        # used during evaluation to avoid unnecessary computation
        self.gated_qs = None
        self.gated_ks = None
        self.gated_vs = None


    def attention(self, query, key, value, attn_mask=None):
        # batch, heads, seqlen, hidden_size
        batch_size, heads, query_len, hidden_size = query.size()
        key_len = key.size(2)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(hidden_size)

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
            attention_scores = attention_scores + attn_mask

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

        output = torch.matmul(attention_weights, value)
        return output


    def forward(self, query, key, value, attn_mask=None):
        # (q, k, v)_seq_len, batch_size, hidden_size 

        if self.training or self.gated_qs is None:
            # recalculate gates values to update them.
            self.gated_qs, self.gated_ks, self.gated_vs = torch.sigmoid(self.query_gate), torch.sigmoid(self.key_gate), torch.sigmoid(self.value_gate)
            self.gated_vs = self.overparameterize(self.gated_vs)

        # apply transformation on query        
        query = self.affine_query(query)
        query = self.ln_query(query)

        # apply gates on all attention inputs
        query, key, value = self.gated_qs * query, self.gated_ks * key, self.gated_vs * value
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        batch_size, query_len, hidden_size = query.size()
        key_len = key.size(1)
        
        query = query.view(batch_size, query_len, 1, self.hidden_size).transpose(1, 2)
        key, value = [ vec.view(batch_size, key_len, 1, self.hidden_size).transpose(1, 2) for vec in [key, value] ]

        output = self.attention(query, key, value, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1)

        return output