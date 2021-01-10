import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstmblock import LSTMBlock
from .adaptive import AdaptiveTiedEmbeddings

class SHALSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, no_layers=4, memory_size=5120, dropouti=0.1, dropouth=0.1, dropouto=0.1):
        super().__init__()

        self.embed_size = embed_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.no_layers = no_layers
        
        self.idrop = nn.Dropout(dropouti)
        self.odrop = nn.Dropout(dropouto)

        # Cutoffs list calculation could be better than this
        start = 4 + int(np.ceil(np.log2(vocab_size / 8) / 2))
        cutoffs = [ 2**x for x in range(start, start + 5, 2) if vocab_size > 2**x ]

        # used as both encoder and decoder (tied weights)
        self.ate = AdaptiveTiedEmbeddings(embed_size, vocab_size, cutoffs)

        self.blocks = nn.ModuleList()
        for idx in range(no_layers):
            # place only one attention head on the layer before the last layer
            self.blocks.append(LSTMBlock(embed_size, hidden_size, dropout=dropouth, use_attn=True if idx == no_layers - 2 else False))

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=1 / np.sqrt(self.embed_size))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden=None, mems=None, targets=None):
        """ Input has shape [seq length, batch] """
        
        # encode and dropout input
        h = self.ate.encode(x)
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
            mem_mask = torch.zeros((len(x), max_mems), device=h.device, dtype=h.dtype)
            attn_mask = torch.cat([mem_mask, attn_mask], dim=-1)

        # iterate over blocks
        new_hidden, new_mems = [], []
        for idx, block in enumerate(self.blocks):
            mem = mems[idx] if mems is not None else None
            hid = hidden[idx] if hidden is not None else None
            h, new_mem, new_hid = block(h, attn_mask, self.memory_size, memory=mem, hidden=hid)
            new_hidden.append(new_hid)
            new_mems.append(new_mem)
        
        # final dropout
        h = self.odrop(h)

        if targets is not None:
            # calculate loss targets are provided
            loss = self.ate(h.view(-1, self.embed_size), targets.view(-1)).loss
            return loss, h, new_hidden, new_mems
        else:
            # calculate predictions
            output = self.ate.predict(h.view(-1, self.embed_size))
            return output, h, new_hidden, new_mems


if __name__ == "__main__":
    model = SHALSTM(200, 256, 512).cuda()
    inp = torch.randint(200, (257, 8)).cuda()
    
    from .optim import MinTrustLamb

    optim = MinTrustLamb(model.parameters(), lr=1e-3)

    model.train()
    new_hidden, new_mems = None, None
    for i in range(1000):
        loss, h, new_hidden, new_mems = model(inp[:-1, :], targets=inp[1:, :])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        print(loss.data)