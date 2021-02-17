
import random
import argparse

import numpy as np
import torch.nn as nn
import torch

from tokenizer import SHALSTMTokenizer
from .model import SHALSTMforQuestionAnswering 


def main(args):
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--val_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)