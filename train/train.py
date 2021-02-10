import time
import math
import random
from glob import glob
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
from torch import autograd

from tensorboardX import SummaryWriter

from shalstm.model import SHALSTM
from shalstm.optim import MinTrustLamb

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(source, step, seq_len):
    
    if len(source) < ( 1 + step + seq_len ):
        seq_len = len(source) - step - 1

    data = source[step:step+seq_len]
    target = source[step + 1:step + 1 + seq_len]
    return data, target


def load_states_from_checkpoint(checkpoint, model=None, optimizer=None, scaler=None):
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])


def evaluate_dir(model, batch_dir, batch_size, set="val", writer=None, ignore_first_batch=False, global_step=0, seq_len=1024, use_amp=True, return_len=False, device=torch.device("cuda")):
    
    print("Evaluating", batch_dir)
    ###
    shift = 0
    ###

    loss, total_len = 0, 0
    hidden, mems = None, None
    for batch in glob(batch_dir):
        l, t, hidden, mems = evaluate(model, batch, batch_size, ignore_first_batch=ignore_first_batch, use_amp=use_amp, hidden=hidden, mems=mems, seq_len=seq_len, device=device)
        loss += l
        total_len += t

    if ignore_first_batch:
        total_len -= seq_len

    loss = loss / total_len
    ppl = math.exp(loss)
    bpt = loss / math.log(2)

    if writer is not None:
        writer.add_scalar(f'{set}/loss', loss, global_step + shift)
        writer.add_scalar(f'{set}/ppl', ppl, global_step + shift)
        writer.add_scalar(f'{set}/bpt', bpt, global_step + shift)

    print("Finished evaluation")

    if return_len:
        return loss, total_len

    return loss


def evaluate(model, batch_path, batch_size, ignore_first_batch=False, hidden=None, mems=None, use_amp=False, seq_len=1024, device=torch.device("cuda")):

    eval_data = torch.load(batch_path).long()
    eval_data = batchify(eval_data, batch_size)

    total_loss, i = 0, 0
    model.eval()
    with torch.no_grad():
        while i < eval_data.size(0) - 2:
            # get minibatch
            data, targets = get_batch(eval_data, i, seq_len=seq_len)
            
            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, output, h, m = model(data, hidden=hidden, mems=mems, targets=targets)
            
            if hidden is not None or not ignore_first_batch:
                total_loss += loss.item() * len(data)

            i += len(data)
            hidden, mems = h, m

    return total_loss, i, hidden, mems


def warmup_lr(optimizer, lr, ratio):
    for param_group in optimizer.param_groups:        
        param_group['lr'] = lr * ratio


def train(
        model,
        batch_path,
        batch_size,
        optimizer,
        scaler,
        base_lr,
        rank=0,
        world_size=1,
        global_step=0,
        seq_len=1024,
        log_interval=100,
        warmup=False,
        writer=None,
        clip_value=-1,
        static_clip=True,
        history_size=100, # this is used only if static_clip is False 
        use_amp=True,
        device=torch.device("cuda")
        ):

    ###
    shift = 0
    ###

    train_data = torch.load(batch_path).long()
    train_data = batchify(train_data, batch_size)
    losses, grad_history = [], []

    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

    total_loss, minibatch, i = 0, 0, 0
    hidden, mems = None, None
    model.train()
    
    with model.join():
        while i < train_data.size(0) - (seq_len + 2):

            # warm up
            ratio = (1 + min(global_step, warmup)) / warmup
            warmup_lr(optimizer, base_lr, min(1.0, ratio))

            # randomly cut out memory %5 of the iterations
            if random.randint(0, max(world_size, 20)) == rank:
                hidden = None
                mems = None

            # get minibatch
            if random.randint(0, max(world_size, 20)) == rank:
                data, targets = get_batch(train_data, i, seq_len=seq_len)
            else:
                data, targets = get_batch(train_data, i, seq_len=seq_len // 2)
            
            # zero out gradients
            model.zero_grad()
            
            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)

            # do backward pass
            scaler.scale(loss).backward()
            
            # debug step
            if model.module._check_nan_grads():
                print("NaN gradients detected, logging...")
                print(f"i={i}, minibatch={minibatch}, batch_path={batch_path}, global_step={global_step}")

            scaler.unscale_(optimizer)

            # calculate dynamic gradient clip threshold
            if not static_clip:
                obs_grad_norm = model.module._get_grad_norm()
                grad_history.append(obs_grad_norm)
                grad_history = grad_history[-history_size:]
                clip_value = np.mean(grad_history)  

            # clip gradients
            if clip_value != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # update weights
            if not model.module._check_nan_grads():
                scaler.step(optimizer)
            
            # update scaler's scale value and other parameters
            scaler.update()

            total_loss += loss.item()
            if minibatch % log_interval == 0 and minibatch > 0:

                cur_loss = min(total_loss / log_interval, 1e1)
                lr = optimizer.param_groups[0]['lr']
                ppl = math.exp(cur_loss)
                bpt = cur_loss / math.log(2)

                if writer is not None:
                    writer.add_scalar('training/loss', cur_loss, global_step + shift)
                    writer.add_scalar('training/scale', scaler.get_scale(), global_step + shift)
                    writer.add_scalar('training/ppl', ppl, global_step + shift)
                    writer.add_scalar('training/bpt', bpt, global_step + shift)
                    writer.add_scalar('training/lr', lr, global_step + shift)

                total_loss = 0

            minibatch += 1
            i += len(data)
            global_step += 1
    
    return global_step


class DummyDDPWrapper(nn.Module):
    def __init__(self, module):
        super(DummyDDPWrapper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def join(self):
        yield


def main(args):

    device = torch.device(args.device)
    use_amp = args.no_amp
    batches = glob(args.train_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model = SHALSTM(args.model_config, device=device)

    # start from checkpoint
    if args.load_checkpoint and rank == 0:
        model.load(args.load_checkpoint)
        print(f"Loaded checkpoint model", args.load_checkpoint)

    model = DummyDDPWrapper(model)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    from apex.optimizers import FusedLAMB
    
    optimizer = FusedLAMB(model.parameters(), lr=args.base_lr,
                 max_grad_norm=args.clip_value, use_nvlamb=True)

    writer = SummaryWriter(args.writer_dir)

    best_loss, global_step = 1e3, 0
    for epoch in range(args.epochs):
        for batch_path in batches:

            # global step is just a counter of batches
            global_step = train(
                model,
                batch_path,
                args.batch_size,
                optimizer,
                scaler,
                args.base_lr,
                global_step=global_step,
                log_interval=args.log_interval,
                seq_len=args.bptt,
                warmup=args.warmup,
                writer=writer,
                use_amp=use_amp,
                device=device
            )

        loss = evaluate_dir(model.module, args.val_dir, args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)

        if best_loss > loss:
            model.module.save(args.checkpoint_path)
        else:
            args.base_lr /= 2

        print(f"Finished epoch {epoch}")

    print("Starting Final Evaluation")
    model.module.load(args.checkpoint_path + ".pt")
    evaluate_dir(model.module, args.val_dir, args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, return_len=True, device=device)
    evaluate_dir(model.module, args.test_dir, args.batch_size, set="test", writer=writer, global_step=global_step, seq_len=args.bptt, return_len=True, device=device)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_lr", type=float, default=2e-3)
    parser.add_argument("--clip_value", type=float, default=0.1)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=25)

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str, default="bin/enwik8/base/base")
    parser.add_argument("--load_checkpoint", type=str, default="")

    parser.add_argument("--model_config", type=str, default="config/base.json")
    parser.add_argument("--writer_dir", type=str, default="runs/enwik8-base")

    args = parser.parse_args()

    print()
    print(args)
    print()

    main(args)