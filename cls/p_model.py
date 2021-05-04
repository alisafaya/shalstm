import torch.nn.functional as F
import torch.nn as nn
import torch

from shalstm import SHALSTM

class SHALSTMforSequenceClassification(SHALSTM):

    def __init__(self, no_labels, config, device=torch.device("cpu")):
        super(SHALSTMforSequenceClassification, self).__init__(config, device)

        self.cls_attn = nn.MultiheadAttention(self.embed_size, 1, dropout=self.dropouth)
        self.attn_param = nn.Parameter(torch.randn(size=(1, 1, self.embed_size), dtype=torch.float, device=device) * 0.01)
        self.projection = nn.Linear(self.embed_size, no_labels)
        self.no_labels = no_labels

        self.cls_attn.apply(self._init_weights)
        self.cls_attn.to(device)
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
        padding_mask = torch.full((batch_size, seq_len), False, device=self.device, dtype=torch.bool)

        for b in range(batch_size):
            mask = torch.where(attention_mask[:, b] == 0) 
            attn_mask[b, :, mask[0]] = -1e6
            attn_mask[b, mask[0], :] = -1e6
            padding_mask[b, mask[0]] = True

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

        logits = self.projection(self.cls_attn(self.attn_param.repeat(1, batch_size, 1), h, h, key_padding_mask=padding_mask)[0].squeeze(0))

        return logits, h, new_hidden, new_mems


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="bin/base/model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    import json
    config = json.loads(open(args.model + ".json").read())
    model = SHALSTMforSequenceClassification(2, config, device=device)
    model.load(args.model + ".pt")
    model.to(device)

    from tokenizer import SHALSTMTokenizer
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)

    inputs = [
        ("A man is playing a large flute.", "A man is playing a flute."),
        ("A woman is not.", "A woman is playing a flute."),
        "A boy is boy.",
    ]
    

    input, attn_mask = tokenizer.encode_as_tensors(inputs)

    loss, h, hidden, mems = model(input, attn_mask)
    output, h, hidden, mems = model(input, attn_mask)

    import ipdb; ipdb.set_trace()
    
    print(loss, output)

    warmup = 5
    total_steps = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    import time
    starttime = time.time()

    model.train()
    for i in range(total_steps):
        model.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, h, hidden, mems = model(input, attn_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        scaler.step(optimizer)
        scaler.update()

    print("Excecution time =", (time.time() - starttime) / total_steps, "sec per batch")

    loss, h, hidden, mems = model(input, attn_mask, labels=torch.tensor([0, 1, 0]))
    output, h, hidden, mems = model(input, attn_mask)

    print(loss, output, output.argmax(dim=1))
