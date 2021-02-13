# coding: utf-8

import argparse

import torch
import torch.nn.functional as F

from shalstm.model import SHALSTM
from tokenizers import Tokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits


def main(args):

    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    device = torch.device(args.device)

    model = SHALSTM.from_pretrained(args.model, device=device)
    model.eval()

    prompt = [args.eos_id,]
    if args.prompt:
        prompt += tokenizer.encode(args.prompt).ids

    sequence = prompt
    prompt = torch.tensor(prompt, dtype=torch.long).view(-1, 1)

    hidden, mems = None, None
    
    if args.prompt:
        output, h,  hidden, mems = model(prompt[:-1], hidden=hidden, mems=mems)
        prompt = prompt[-1:]

    with torch.no_grad():
        for i in range(args.max_length):
            output, h, hidden, mems = model(prompt, hidden=hidden, mems=mems)

            if args.use_sampling:
                output = top_k_top_p_filtering(output.view(-1) / args.temperature, top_p=args.top_p).view(*output.shape)
                token_weights = F.softmax(output, dim=-1).squeeze()
                output_idx = torch.multinomial(token_weights, num_samples=1)[0]
            else:
                output_idx = torch.argmax(output.view(-1))

            prompt.fill_(output_idx)
            sequence.append(output_idx.item())
            if output_idx == args.eos_id:
                break

    print(tokenizer.decode(sequence))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--eos_id", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()
    main(args)