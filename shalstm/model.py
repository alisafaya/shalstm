import math
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstmblock import LSTMBlock
from .adaptive import AdaptiveTiedEmbeddings
from .splitsoftmax import SplitSoftmaxWithLoss
from .utils import top_k_top_p_filtering

class SHALSTM(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super(SHALSTM, self).__init__()
        self.load_config(config)

        self.idrop = nn.Dropout(self.dropouti)
        self.odrop = nn.Dropout(self.dropouto)

        # Cutoffs list calculation could be better than this
        if self.vocab_size > 1024:
            start = 4 + int(np.ceil(np.log2(self.vocab_size / 8) / 2))
            cutoffs = [ 2**x for x in range(start, start + 5, 2) if self.vocab_size > 2**x ]
        else:
            cutoffs = []

        # used as both encoder and decoder (tied weights)
        self.encoder = nn.Embedding(self.vocab_size, self.embed_size)
        self.splitloss = SplitSoftmaxWithLoss(self.embed_size, self.vocab_size, cutoffs, self.encoder.weight).to(device)

        self.blocks = nn.ModuleList()
        for idx in range(1, self.no_layers + 1):
            # place only one attention head on the layer before the last layer
            self.blocks.append(LSTMBlock(self.embed_size, self.hidden_size, rnn=self.rnn_type, dropout=self.dropouth, use_attn=idx in self.attn_layers, device=device))

        self.apply(self.init_weights)

        self.device = device
        self.to(device)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.embed_size))

        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def load_config(self, config):
        if isinstance(config, dict):
            self.embed_size = config.get("embed_size", 1024)
            self.memory_size = config.get("memory_size", 4096)
            self.hidden_size = config.get("hidden_size", 4096)
            self.no_layers = config.get("no_layers", 4)
            self.dropouti = config.get("dropouti", 0.1)
            self.dropouto = config.get("dropouto", 0.1)
            self.dropouth = config.get("dropouth", 0.1)
            self.vocab_size = config.get("vocab_size", 2**14)
            self.attn_layers = config.get("attn_layers", [3])
            self.rnn_type = config.get("rnn_type", "lstm")
        elif isinstance(config, str):
            config = json.loads(open(config).read())
            self.load_config(config)
        else:
            raise TypeError

    def get_config(self):
        config = {
            "embed_size": self.embed_size,
            "memory_size": self.memory_size,
            "hidden_size": self.hidden_size,
            "no_layers": self.no_layers,
            "dropouti": self.dropouti,
            "dropouto": self.dropouto,
            "dropouth": self.dropouth,
            "vocab_size": self.vocab_size,
            "attn_layers": self.attn_layers,
            "rnn_type": self.rnn_type
        }
        return config

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def _check_nan_grads(self):
        for p in self.parameters():
            if p.grad is not None:
                if (p.grad.data != p.grad.data).any():
                    return True

        return False 

    def forward(self, x, hidden=None, mems=None, targets=None, attention_mask=None):
        """ Input has shape [seq length, batch] """
        x = x.to(self.device)
        seq_len, batch_size = x.shape

        # encode and dropout input
        h = self.encoder(x)
        h = self.idrop(h)

        # if memory is provided, trim it to fit max memory size
        if mems is not None:
            maxmem = self.memory_size - len(h)
            mems = [m[-maxmem:] for m in mems]
        total_length = len(x) + (len(mems[0]) if mems else 0)

        # create attention mask
        if attention_mask is None:
            attn_mask = torch.full((seq_len, seq_len), -1e6, device=h.device, dtype=h.dtype) # instead of -Inf we use -1,000,000
            attn_mask = torch.triu(attn_mask, diagonal=1)
            
            # concatenate memories from the previous pass if provided
            if mems is not None:
                max_mems = max(len(m) for m in mems)
                mem_mask = torch.zeros((seq_len, max_mems), device=h.device, dtype=h.dtype)
                attn_mask = torch.cat([mem_mask, attn_mask], dim=-1)
        
        else:
            attention_mask = attention_mask.to(self.device)
            attn_mask = torch.full((batch_size, seq_len, seq_len), -1e6, device=self.device, dtype=h.dtype)
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

        if targets is not None:
            # calculate loss targets are provided
            loss = self.splitloss(h.view(-1, self.embed_size), targets.to(self.device).view(-1)).loss
            return loss, h, new_hidden, new_mems
        else:
            # calculate predictions
            output = self.splitloss.log_prob(h.view(-1, self.embed_size))
            output = output.view(x.size(0), -1)
            return output, h, new_hidden, new_mems

    def generate(self, eos_id=2, initial_prompt=None, max_length=1024, use_sampling=True, top_p=0.95, temperature=1.0):
        """ initial_prompt sequence has shape [seq length] """

        sequence = [] if initial_prompt is None else initial_prompt
        if initial_prompt is not None:
            prompt = [eos_id,] + initial_prompt
        else:
            prompt = [eos_id,]
        prompt = torch.tensor(prompt, dtype=torch.long).view(-1, 1)

        self.eval()
        hidden, mems = None, None
        with torch.no_grad():
            if initial_prompt is not None:
                output, h,  hidden, mems = self(prompt[:-1], hidden=hidden, mems=mems)
                prompt = prompt[-1:]

            for i in range(max_length):
                
                output, h, hidden, mems = self(prompt, hidden=hidden, mems=mems)
                if use_sampling:
                    token_weights = top_k_top_p_filtering(torch.exp(output.view(-1)) / temperature, top_p=top_p, filter_value=0.0)
                    output_idx = torch.multinomial(token_weights, num_samples=1)[0]
                else:
                    output_idx = torch.argmax(output.view(-1))

                prompt.fill_(output_idx)
                sequence.append(output_idx.item())
                if output_idx == eos_id:
                    break

        return sequence

    def save(self, path):
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.cpu()

        with open(path + ".json", "w") as fo:
            fo.write(json.dumps(self.get_config(), indent=4))
        torch.save(state_dict, path + ".pt")

    def load(self, path, strict=False):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(cls, path, device=torch.device("cpu")):
        config = json.loads(open(path + ".json").read())
        model = cls(config, device=device)
        model.load(path + ".pt")
        model.to(device)

        return model


if __name__ == "__main__":
    device = torch.device("cuda")
    use_amp = False

    model_config = json.loads(open("config/base_exp.json").read())
    model = SHALSTM(model_config, device=device)
    print("No of parameters", sum(p.numel() for p in model.parameters()))

    inp = torch.load(f"/userfiles/asafaya19/pile/train/batch_00001.pt")[:1025*16].long().view(1025, 16).to(device) # input size x batch size

    from .optim import MinTrustLamb

    optimizer = MinTrustLamb(model.parameters(), lr=1e-3)

    model.train()

    import time
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hidden, mems = None, None
    starttime = time.time()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=4, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profiling"),
        ) as profiler:

        for i in range(20):

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, h, hidden, mems = model(inp[:-1, :], targets=inp[1:, :], hidden=hidden, mems=mems)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            scaler.step(optimizer)
            scaler.update()
#             print(loss.item())
            profiler.step()

    print("Excecution time =", (time.time() - starttime) / 20, "sec per batch")

#     model.save("bin/sample/sample_model")
#     new_model = SHALSTM.from_pretrained("bin/sample/sample_model", device="cuda:0")
#     loss, h, new_hidden, new_mems = model(inp[:-1, :], targets=inp[1:, :])
#     print(loss.data.item())
    print("No of parameters", sum(p.numel() for p in model.parameters()))
