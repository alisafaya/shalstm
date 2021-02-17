# coding: utf-8

import argparse

import torch
import torch.nn.functional as F

from shalstm.model import SHALSTM
from shalstm.utils import top_k_top_p_filtering
from tokenizers import Tokenizer

def main(args):

    device = torch.device(args.device)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = SHALSTM.from_pretrained(args.model, device=device)

    prompt = None
    if args.prompt:
        prompt = tokenizer.encode(args.prompt, add_special_tokens=False).ids

    sequence = model.generate(eos_id=args.eos_id, initial_prompt=prompt, max_length=args.max_length, use_sampling=args.use_sampling, temperature=args.temperature, top_p=args.top_p)

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