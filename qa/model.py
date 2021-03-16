import torch.nn as nn
import torch

from shalstm import SHALSTM
from shalstm.utils import top_k_top_p_filtering

class SHALSTMforQuestionAnswering(SHALSTM):

    def forward(self, input, attention_mask=None, type_ids=None, hidden=None, mems=None, return_loss=False, lm_loss=False):
        """
        all arguments have shape (seq length, batch)
        padding should be on left for input, on right for targets (as in seq2seq models)
    
            - type_ids is used both for loss masking and attention masking. it should be 1 for the answer tokens and 0 otherwise. 
            - attention_mask (attention mask) is 0 for paddings and 1 for other tokens 
        """
        x = input[:-1].to(self.device)
        targets = input[1:].to(self.device)

        seq_len, batch_size = x.shape

        if attention_mask is None:
            attention_mask = torch.ones(*x.shape)

        if type_ids is None:
            type_ids = torch.zeros(*input.shape)
        loss_mask = type_ids[1:].view(-1).to(self.device)

        # encode and dropout input
        h = self.ate.encode(x)
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
            mask = torch.where(attention_mask[:-1, b] == 0) 
            attn_mask[b, :, mask[0]] = -1e6
            attn_mask[b, mask[0], :] = -1e6

        # concatenate memories from the previous pass if provided
        if mems is not None:
            max_mems = max(len(m) for m in mems)
            mem_mask = torch.zeros((batch_size, seq_len, max_mems), device=h.device, dtype=h.dtype)
            attn_mask = torch.cat([mem_mask, attn_mask], dim=2)

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

        if return_loss:
            if not lm_loss:
                # calculate loss targets are provided
                loss = -(self.ate(h.view(-1, self.embed_size), input[1:].to(self.device).view(-1)).output * loss_mask).mean() # .view(*x.shape).mean(0).mean()
            else:
                # calculate loss on all tokens
                loss = self.ate(h.view(-1, self.embed_size), input[1:].to(self.device).view(-1)).loss
            return loss, h, new_hidden, new_mems
        else:
            # calculate predictions
            output = self.ate.log_prob(h.view(-1, self.embed_size))
            output = output.view(*x.shape, -1)
            return output, h, new_hidden, new_mems

    def conditional_generate(self, input, attention_mask, type_ids, eos_id=2, max_length=64, use_sampling=False, top_p=0.95, temperature=1.0):
        """ input sequence has shape [seq length, batch size] """

        prompt = torch.cat([input, torch.zeros(1, input.shape[1], dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.ones(1, attention_mask.shape[1])])
        type_ids = torch.cat([type_ids, torch.zeros(1, type_ids.shape[1])])

        self.eval()
        sequences = torch.zeros(max_length, input.shape[1], dtype=torch.long)
        hidden, mems = None, None
        with torch.no_grad():
            output, h,  hidden, mems = self(prompt[:-1], attention_mask=attention_mask[:-1], type_ids=type_ids[:-1], hidden=hidden, mems=mems)
            
            prompt = prompt[-2:]
            attention_mask=attention_mask[-2:]
            type_ids=type_ids[-2:]

            for i in range(max_length):
                output, h,  hidden, mems = self(prompt, attention_mask=attention_mask, type_ids=type_ids, hidden=hidden, mems=mems)

                if use_sampling:
                    raise NotImplementedError
                    token_weights = top_k_top_p_filtering(torch.exp(output.view(-1)) / temperature, top_p=top_p, filter_value=0.0)
                    output_idx = torch.multinomial(token_weights, num_samples=1)[0]
                
                else:
                    output_idx = torch.argmax(output, dim=-1)

                prompt[0, :] = output_idx
                sequences[i, :] = output_idx

                if torch.all(output_idx == eos_id):
                    break

        return sequences

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="bin/base/model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = SHALSTMforQuestionAnswering.from_pretrained(args.model, device=torch.device(args.device))

    from tokenizer import SHALSTMTokenizer
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)

    questions = [
        "another thing there",
        "some length here",
    ]

    answers = [
        "brother Hi how",
        "this answer for question one",
    ]

    input, attn_mask, type_ids, input_length = tokenizer.encode_for_qa(questions, answers)

    loss, h, hidden, mems = model(input, attn_mask, type_ids, return_loss=True)

    warmup = 5
    total_steps = 150
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: float(x / warmup) if x < warmup else float((total_steps - x) / total_steps)])
    
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    import time
    starttime = time.time()

    model.train()
    for i in range(total_steps):

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, h, hidden, mems = model(input, attn_mask, type_ids, return_loss=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if i % (total_steps // 10) == 0:
            print(loss.item())

    print("Excecution time =", (time.time() - starttime) / total_steps, "sec per batch")

    questions = [
        "question one ?",
        "some length here",
    ]

    answers = [
        "this answer to this one",
        "This is another answer for another question ",
    ]

    input, attn_mask, type_ids, input_length = tokenizer.encode_for_qa(questions, answers)
    
    with torch.no_grad():
        model.eval()
        output, h, hidden, mems = model(input, attn_mask, type_ids)
        output = torch.argmax(output, dim=-1)

    ids = output[input_length - 1:].t().cpu().tolist()
    print(tokenizer.decode(ids[0]))
    print(tokenizer.decode(ids[1]))

    sequence = model.conditional_generate(tokenizer.encode("another thing there").ids, max_length=10, use_sampling=False)
    
    print("Conditional generation")
    print(tokenizer.decode(sequence))

# Assigning bigger loss to longer sequences

#     In [54]: for x in attn_mask[0]:
#     ...:     print("".join([ f"{y:10.1f}" for y in x.tolist()]))
#     ...:

# -1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0-1000000.0
# -1000000.0-1000000.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0-1000000.0
# -1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0-1000000.0