import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, FForwardNetwork

class LSTMBlock(nn.Module):
    """LSTM Block (lstm -> attention (optional) -> feedforward network)"""
    def __init__(self, input_size, fforward_size, rnn="lstm", dropout=0.1, use_attn=False, device=torch.device("cpu")):
        super(LSTMBlock, self).__init__()
        
        if rnn == "lstm":
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size)
        elif rnn == "gru":
            self.lstm = nn.GRU(input_size=input_size, hidden_size=input_size)
        elif rnn == "sru":
            from sru import SRU
            self.lstm = SRU(input_size=input_size, hidden_size=input_size, num_layers=2, amp_recurrence_fp16=True)
        else:
            raise TypeError("rnn type should be one of lstm, gru, sru")

        self.attn = Attention(input_size, dropout=dropout, device=device) if use_attn else None
        self.fforward = FForwardNetwork(input_size, fforward_size, dropout=dropout, device=device)

        self.ln_input = nn.LayerNorm(input_size, eps=1e-12)
        self.ln_queries = nn.LayerNorm(input_size, eps=1e-12) if use_attn else None
        self.ln_values = nn.LayerNorm(input_size, eps=1e-12) if use_attn else None
        self.ln_fforward = nn.LayerNorm(input_size, eps=1e-12)
        self.ln_output = nn.LayerNorm(input_size, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.device = device
        self.to(device)

    def forward(self, h, attn_mask, memory_size, memory=None, hidden=None):
        # seq_len, batch_size, hidden_size
        h = h.to(self.device)

        # normalize the input of the block
        h = self.ln_input(h)

        # pass through rnn and detach hidden
        if type(self.lstm) == nn.LSTM:
            lstm_output, lstm_hidden = self.lstm(h, None if hidden is None else tuple(x.to(self.device) for x in hidden))
            new_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())
        else:
            lstm_output, lstm_hidden = self.lstm(h, None if hidden is None else hidden.to(self.device))
            new_hidden = lstm_hidden.detach()
        
        # lstm output dropout
        if self.dropout is not None:
            lstm_output = self.dropout(lstm_output)
        
        # lstm residual
        h = h + lstm_output

        new_memory = []
        if self.attn is not None:
            values = self.ln_values(h)
            
            if memory is not None:
                # if previous memory exists, concatenate it
                values = torch.cat([memory.to(self.device), values], dim=0)
            
            h = self.ln_queries(h)
            queries = h
            keys = values
            
            # apply attention 
            attn_output = self.attn(queries, keys, values, attn_mask)

            # attention output dropout
            if self.dropout is not None:
                attn_output = self.dropout(attn_output)

            # store memory for future steps
            new_memory = values[-memory_size:].detach()

            # attention residual
            h = h + attn_output

        # feed forward
        x = self.ln_fforward(h)
        x = self.fforward(x)
    
        # apply dropout on feed forward network output
        if self.dropout is not None:
            x = self.dropout(x)

        h = self.ln_output(h)
        h = x + h

        # assert not h.isnan().any()

        return h, new_memory, new_hidden
