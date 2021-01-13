import argparse
import functools
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import data
from shalstm.model import SHALSTM
from shalstm.optim import MinTrustLamb

from utils import batchify, get_batch

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--memsize', type=int, default=5120,
                    help='size of attention memory')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='number of hidden units for feedforward network')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='lr decay value')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=1024,
                    help='sequence length')
parser.add_argument('--warmup', type=int, default=1000,
                    help='warmup for learning rate')
parser.add_argument('--cooldown', type=int, default=None,
                    help='cooldown for learning rate')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.1,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--static_clip', action='store_true',
                    help='use dynamic gradient clipping')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save(model, f)

def model_load(fn):
    global model, optimizer
    with open(fn, 'rb') as f:
        m = torch.load(f)
        d = m.state_dict()
        model.load_state_dict(d, strict=False)

import os
import hashlib

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = min(100, args.batch_size)
print('Eval batch size of', eval_batch_size)
test_batch_size = 8
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print('Total number of tokens:', ntokens)

model = SHALSTM(ntokens, args.emsize, args.hidden_size, args.nlayers, args.memsize, args.dropouth, args.dropouti, args.dropout)

if args.resume and args.epochs > 0:
    print('Resuming model ...')
    model_load(args.resume)
    model.dropouti, model.dropouth, model.dropout = args.dropouti, args.dropouth, args.dropout

if args.cuda:
    model = model.cuda()

params = list(model.parameters())
total_params = sum(x.numel() for x in params)
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = None
    mems = None
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            raw_loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)
            total_loss += len(data) * raw_loss.data
            
    return total_loss.item() / len(data_source)

grad_history = []
history_size = 100

def train(epoch=0):
    # Turn on training mode which enables dropout.
    global grad_history

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = None
    mems = None
    batch, i = 0, 0
    losses = []
    clip_value = args.clip
    
    while i < train_data.size(0) - 1 - 1:

        # Warmup
        for param_group in optimizer.param_groups:
            step = epoch * (len(train_data) // args.bptt) + batch + 1
            pctwarm = min(step, args.warmup) / args.warmup
            if pctwarm < 1:
                param_group['lr'] = args.lr * pctwarm
        
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, args.bptt)

        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        
        if not args.static_clip:
            obs_grad_norm = model._get_grad_norm()
            grad_history.append(obs_grad_norm)
            grad_history = grad_history[-history_size:]
            clip_value = np.mean(grad_history)  

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | clip {:05.3f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch + 1, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'], clip_value,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

        batch += 1
        i += seq_len

# At any point you can hit Ctrl + C to break out of training early.
try:
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000
    optimizer = MinTrustLamb(params, lr=args.lr, weight_decay=args.wdecay, min_trust=0.25)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Loop over epochs.
    for epoch in range(1, args.epochs+1):

        epoch_start_time = time.time()
        train(epoch - 1)
        val_loss = evaluate(val_data, eval_batch_size)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.3f}s | valid loss {:5.3f} | '
            'valid ppl {:8.3f} | valid bpc {:8.3f}'.format(
          epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)

        if val_loss < stored_loss:
            model_save(args.save)
            print('Saving model (new best validation)')
            stored_loss = val_loss

        best_val_loss.append(val_loss)
        lr_scheduler.step()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Model total parameters:', total_params)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.3f} | test ppl {:8.3f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
