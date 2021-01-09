import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention

class FForwardNetwork(nn.Module):
    """Feed forward network or Boom layer as Smerity names it"""
    def __init__(self, input_size, feedforward_size=None, dropout=0.1, activation=nn.GELU()):
        super(FForwardNetwork, self).__init__()
        
        feedforward_size = input_size * 2 if feedforward_size is None else feedforward_size

        self.linear1 = nn.Linear(input_size, feedforward_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else False
        self.linear2 = nn.Linear(feedforward_size, input_size)
        self.activation = activation

    def forward(self, x):
        
        x = self.linear1(x)
        x = self.activation(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.activation(x)
        return x


class LSTMBlock(nn.Module):
    """LSTM Block (lstm -> attention (optional) -> feedforward network)"""
    def __init__(self, input_size, fforward_size, dropout=0.1, use_attn=False):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, batch_first=False)
        self.attn = Attention(input_size) if use_attn else None
        self.fforward = FForwardNetwork(input_size, fforward_size, dropout=dropout)

        self.ln_input = nn.LayerNorm(input_size, eps=1e-12)
        self.ln_queries = nn.LayerNorm(input_size, eps=1e-12) if use_attn else None
        self.ln_values = nn.LayerNorm(input_size, eps=1e-12) if use_attn else None
        self.ln_fforward = nn.LayerNorm(input_size, eps=1e-12)
        self.ln_output = nn.LayerNorm(input_size, eps=1e-12)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, h, attn_mask, memory_size, memory=None, hidden=None):
        # seq_len, batch_size, hidden_size

        # normalize the input of the block
        h = self.ln_input(h)
        
        # pass through lstm
        lstm_output, lstm_hidden = self.lstm(h, None if hidden is None else hidden)
        
        # detach hidden
        new_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())
        
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
                values = torch.cat([memory, values], dim=0)
            
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

        return h, new_memory, new_hidden
