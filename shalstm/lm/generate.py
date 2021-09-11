import argparse

import torch
import torch.nn.functional as F

from ..model import SHALSTM
from ..utils import top_k_top_p_filtering
from transformers import PreTrainedTokenizerFast

def main(args):

    device = torch.device(args.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = SHALSTM.from_pretrained(args.model, device=device)

    prompt = None
    if args.prompt_file:
        prompt = tokenizer.encode(open(args.prompt_file).read(), add_special_tokens=False)
    elif args.prompt:
        prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        prompt = []

    print("Encoded", len(prompt), "Tokens")
    sequence = model.generate(eos_id=args.eos_id, initial_prompt=prompt, max_length=args.max_length, use_sampling=args.use_sampling, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    print(tokenizer.decode(sequence, skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--eos-id", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.975)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--use-sampling", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    args = parser.parse_args()
    main(args)