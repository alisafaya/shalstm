import torch.nn.functional as F
import torch.nn as nn
import torch

from ..model import SHALSTM

class SHALSTMforSequenceClassification(SHALSTM):

    def __init__(self, no_labels, config, device=torch.device("cpu")):
        super(SHALSTMforSequenceClassification, self).__init__(config, device)

        self.projection = nn.Linear(self.embed_size, no_labels)
        self.projection.apply(self._init_weights)
        self.projection.to(device)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, input, attention_mask=None, hidden=None, mems=None):
        """
        all arguments have shape (seq length, batch)
        padding should be on right.
    
            - attention_mask (attention mask) is 0 for paddings and 1 for other tokens 
        """
        x = input.to(self.device)
        seq_len, batch_size = x.shape

        if attention_mask is None:
            attention_mask = torch.ones(*x.shape)

        # get the last non-padding token index
        last_token_index = attention_mask.sum(dim=0) - 1

        # encode and dropout input
        h = self.encoder(x)
        h = self.idrop(h)

        # if memory is provided, trim it to fit max memory size
        if mems is not None:
            maxmem = self.memory_size - len(h)
            mems = [m[-maxmem:] for m in mems]
        total_length = len(x) + (len(mems[0]) if mems else 0)

        # construct attention mask:
        attn_mask = torch.full((batch_size, seq_len, seq_len), -1e6, device=self.device, dtype=h.dtype) # instead of -Inf we use -1,000,000
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for b in range(batch_size):
            mask = torch.where(attention_mask[:, b] == 0) 
            attn_mask[b, :, mask[0]] = -1e6
            attn_mask[b, mask[0], :] = -1e6

        # concatenate memories from the previous pass if provided
        if mems is not None:
            max_mems = max(len(m) for m in mems)
            mem_mask = torch.zeros((batch_size, seq_len, max_mems), device=h.device, dtype=h.dtype)
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
        last_token_hidden = h[last_token_index, torch.arange(batch_size), :]
        logits = self.projection(last_token_hidden)

        return logits, h, new_hidden, new_mems
