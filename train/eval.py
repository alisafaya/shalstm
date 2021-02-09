import os
import math
import argparse
from glob import glob

import torch
import torch.distributed as dist
import torch.nn as nn

from shalstm.model import SHALSTM
from .train import evaluate_dir

def main(args):
    device = torch.device(args.device)
    print("Using device:", device)
    print("ignore_first_batch:", args.ignore_first_batch)

    model = SHALSTM.from_pretrained(args.model, device=device)
    loss, length = evaluate_dir(model, args.data_dir, args.batch_size, ignore_first_batch=args.ignore_first_batch, seq_len=args.bptt, return_len=True, device=device)

    print('Unnormalized evaluation results')
    print('| loss {:5.3f} | ppl {:8.3f} | bpt {:8.3f}'.format(loss, math.exp(loss), loss / math.log(2)))

    if args.data_length != -1:
        print("Normalizing loss")
        print("Tokenized data length =", args.batch_size * length)
        print("Original data length =", args.data_length)
        
        loss = (loss * args.batch_size * length) / args.data_length
        print('Normalized evaluation results')
        print('| loss {:5.3f} | ppl {:8.3f} | bpt {:8.3f}'.format(loss, math.exp(loss), loss / math.log(2)))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--ignore_first_batch", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_length", type=int, default=-1)
    args = parser.parse_args()
    main(args)