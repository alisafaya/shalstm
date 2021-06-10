import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import Attention, FForwardNetwork
from torch.nn import LayerNorm as LayerNorm

class LSTMBlock(nn.Module):
    """LSTM Block (lstm -> attention (optional) -> feedforward network)"""
    def __init__(self, input_size, fforward_size, rnn="lstm", dropout=0.1, use_attn=False, device=torch.device("cpu")):
        super(LSTMBlock, self).__init__()

        if rnn == "lstm":
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size)
            self.rnn_init = tuple(nn.Parameter(torch.randn(1, 1, input_size) / math.sqrt(input_size)) for x in range(2))
        elif rnn == "gru":
            self.lstm = nn.GRU(input_size=input_size, hidden_size=input_size)
            self.rnn_init = nn.Parameter(torch.randn(1, 1, input_size) / math.sqrt(input_size))
        elif rnn == "sru":
            from sru import SRU
            self.lstm = SRU(input_size=input_size, hidden_size=input_size, num_layers=2, amp_recurrence_fp16=True, layer_norm=True)
            self.rnn_init = nn.Parameter(torch.randn(2, 1, input_size) / math.sqrt(input_size))
        else:
            raise TypeError("rnn type should be one of lstm, gru, sru")
        
        self.ln_h = LayerNorm(input_size, eps=1e-12)

        self.attn_hidden = input_size
        self.init_memsize = int(2 ** math.ceil(math.log2(math.sqrt(input_size))))
        self.attn = Attention(self.attn_hidden, dropout=dropout, device=device) if use_attn else None
        self.attn_init = nn.Parameter(torch.randn(self.init_memsize, 1, input_size) / math.sqrt(self.attn_hidden)) if use_attn else None

        self.affine_queries = nn.Linear(input_size, self.attn_hidden) if use_attn else None
        self.ln_queries = LayerNorm(self.attn_hidden, eps=1e-12) if use_attn else None
        self.ln_values = LayerNorm(self.attn_hidden, eps=1e-12) if use_attn else None

#         self.affine_values = nn.Linear(input_size, self.attn_hidden) if use_attn else None
#         self.affine_attn_out = nn.Linear(self.attn_hidden, input_size) if use_attn else None
#         self.ln_attn_out = LayerNorm(input_size, eps=1e-12) if use_attn else None

        self.fforward = FForwardNetwork(input_size, fforward_size, dropout=dropout, device=device)
        self.ln_fforward = LayerNorm(input_size, eps=1e-12)
        self.ln_output = LayerNorm(input_size, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.device = device
        self.to(device)

    def forward(self, input, attn_mask, memory_size, memory=None, hidden=None):
        # seq_len, batch_size, hidden_size
        seq_len, batch_size, hidden_size = input.shape
        h = input.to(self.device)

        # pass through rnn and detach hidden
        if type(self.lstm) == nn.LSTM:
            lstm_hidden = tuple(x.repeat(1, batch_size, 1).to(self.device) for x in self.rnn_init) if hidden is None else tuple(x.to(self.device) for x in hidden)

            h, lstm_hidden = self.lstm(h, lstm_hidden)
            new_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())

        else:
            rnn_hidden = self.rnn_init.repeat(1, batch_size, 1).to(self.device) if hidden is None else hidden.to(self.device)
            h, rnn_hidden = self.lstm(h, rnn_hidden)
            new_hidden = rnn_hidden.detach()
        
        # lstm output dropout
        if self.dropout is not None:
            h = self.dropout(h)
        
        new_memory = []
        if self.attn is not None:
            h = self.ln_h(h)

            values = self.ln_values(h)
            if memory is not None:
                # if previous memory exists, concatenate it
                values = torch.cat([ memory.to(self.device), values], dim=0)
            else:
                mem_mask_shape = attn_mask.shape[:-1] + (self.init_memsize,)
                mem_mask = torch.zeros(mem_mask_shape, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([mem_mask, attn_mask], dim=-1)

                values = torch.cat([ self.attn_init.to(self.device).repeat(1, batch_size, 1), values], dim=0)
            keys = values

            queries = self.affine_queries(h)
            queries = self.ln_queries(queries)

            # apply attention 
            attn_output = self.attn(queries, keys, values, attn_mask)
#             attn_output = self.ln_attn_out(self.affine_attn_out(attn_output))

            # attention output dropout
            if self.dropout is not None:
                attn_output = self.dropout(attn_output)

            # store memory for future steps
            new_memory = values[-memory_size:].detach()

            # attention residual
            h = h + attn_output
        
        # feed forward
        h = self.ln_fforward(h)
        h = h + self.fforward(h) + input
        h = self.ln_output(h)

        return h, new_memory, new_hidden
