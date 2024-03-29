import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm

class FForwardNetwork(nn.Module):
    """Feed forward network or Boom layer as Smerity names it"""
    def __init__(self, input_size, feedforward_size=None, shortcut=True, dropout=0.1, activation=nn.GELU(), device=torch.device("cpu")):
        super(FForwardNetwork, self).__init__()

        self.shortcut = shortcut
        self.input_size = input_size
        self.feedforward_size = input_size * (4 if shortcut else 2) if feedforward_size is None else feedforward_size

        self.linear1 = nn.Linear(input_size, self.feedforward_size, bias=False)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else False

        self.linear2 = nn.Linear(self.feedforward_size, input_size, bias=False) if not shortcut else False

        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.linear1(x)
        x = self.activation(x)

        if self.dropout:
                x = self.dropout(x)
        
        if self.shortcut:
            x = torch.narrow(x, -1, 0, self.feedforward_size)
            x = x.view(*x.shape[:-1], self.feedforward_size // self.input_size, self.input_size)
            x = x.sum(dim=-2)
        else:
            x = self.linear2(x)

        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, device=torch.device("cpu")):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale_val = math.sqrt(hidden_size)

        # gates
        self.query_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.key_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        self.value_gate = nn.Parameter(torch.zeros(size=(1, 1, hidden_size), dtype=torch.float))
        
        # over parameterized values gate
        self.overparameterize = FForwardNetwork(hidden_size, feedforward_size=hidden_size * 2, dropout=dropout, activation=nn.Tanh(), device=device)

        # used during evaluation to avoid unnecessary computation
        self.gated_qs = None
        self.gated_ks = None
        self.gated_vs = None

        self.device = device
        self.to(device)


    def attention(self, query, key, value, attn_mask=None):
        # batch, heads, seqlen, hidden_size
        batch_size, heads, query_len, hidden_size = query.size()
        key_len = key.size(2)
        
        attention_scores = torch.matmul(query / self.scale_val, key.transpose(-1, -2).contiguous())

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, *attn_mask.shape[-2:])
            else:
                attn_mask = attn_mask.view(batch_size, 1, *attn_mask.shape[-2:])

            attention_scores = attention_scores + attn_mask

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, heads, query_len, key_len)

        output = torch.matmul(attention_weights, value)
        return output


    def forward(self, query, key, value, attn_mask=None):

        query, key, value, attn_mask = query.to(self.device), key.to(self.device), value.to(self.device), attn_mask.to(self.device)

        # (q, k, v)_seq_len, batch_size, hidden_size 
        if self.training or self.gated_qs is None:
            # recalculate gates values to update them.
            self.gated_qs, self.gated_ks, self.gated_vs = torch.sigmoid(self.query_gate), torch.sigmoid(self.key_gate), torch.sigmoid(self.value_gate)
            self.gated_vs = self.overparameterize(self.gated_vs)

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