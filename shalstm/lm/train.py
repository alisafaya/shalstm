import time
import math
import random
from glob import glob
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
from torch import autograd
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from tensorboardX import SummaryWriter

from ..model import SHALSTM
from ..optim import MinTrustLamb

class DummyDDPWrapper(nn.Module):
    def __init__(self, module):
        super(DummyDDPWrapper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @contextmanager
    def join(self):
        yield


def load_states_from_checkpoint(checkpoint, model=None, optimizer=None, scaler=None, scheduler=None):
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])


def evaluate(model, batch_path, use_amp=False, seq_len=1024):

    eval_data = torch.load(batch_path)

    model.eval()
    total_loss = 0
    total_length = 0

    with torch.no_grad():
        for inp_ids, attn_masks in eval_data:
            
            hidden, mems = None, None
            for i in range(0, inp_ids.size(0) - 1, seq_len):

                if inp_ids.size(0) < ( 1 + i + seq_len ):
                    step_len = inp_ids.size(0) - i - 1
                else:
                    step_len = seq_len

                # get minibatch
                data = inp_ids[i:i+step_len].long()
                attention_mask = attn_masks[i:i+step_len].long()
                targets = inp_ids[i+1:i+step_len+1].long()

                # calculate loss
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets, attention_mask=attention_mask)

                total_loss += loss.item() * len(data)
                total_length += len(data)
                i += len(data)

    return total_loss, total_length


def evaluate_dir(model, batch_dir, set="val", writer=None, global_step=0, seq_len=1024, use_amp=False):

    print("Global Step =", global_step, "Evaluating", batch_dir)

    loss, total_len = 0, 0
    for batch in glob(batch_dir):
        l, t = evaluate(model, batch, use_amp=use_amp, seq_len=seq_len)
        loss += l
        total_len += t

    loss = loss / total_len
    ppl = math.exp(loss)
    bpt = loss / math.log(2)

    if writer is not None:
        writer.add_scalar(f'{set}/loss', loss, global_step)
        writer.add_scalar(f'{set}/ppl', ppl, global_step)
        writer.add_scalar(f'{set}/bpt', bpt, global_step)
        print(f'| step {global_step:0>6} | loss {loss:5.2f} |')

    print("Finished evaluation")

    return loss


def train(
        model,
        batch_path,
        optimizer,
        scaler,
        val_dir=None,
        global_step=0,
        eval_steps=1e10,
        max_steps=1e10,
        seq_len=1024,
        log_interval=100,
        writer=None,
        clip_value=-1,
        static_clip=True,
        history_size=100, # this is used only if static_clip is False 
        use_amp=True,
        lr_scheduler=None,
        device=torch.device("cuda"),
        checkpoint_path="shalstm",
        rank=0,
        dist=None
        ):

    train_data = torch.load(batch_path)

    model.train()
    losses, grad_history = [], []
    total_loss, minibatch, obs_grad_norm = 0, 0, 0
    with model.join(divide_by_initial_world_size=False):
        for inp_ids, attn_masks in train_data:
            hidden, mems = None, None
            for i in range(0, inp_ids.size(0) - 1, seq_len):
                # get minibatch
                if inp_ids.size(0) < ( 1 + i + seq_len ):
                    step_len = inp_ids.size(0) - i - 1
                else:
                    step_len = seq_len
                    
                if step_len < 5:
                    continue

                data = inp_ids[i:i+step_len].long()
                attention_mask = attn_masks[i:i+step_len].long()
                targets = inp_ids[i+1:i+step_len+1].long()

                # zero out gradients
                model.zero_grad()

                # calculate loss
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets, attention_mask=attention_mask, scale_loss_bptt=True)

                # do backward pass
                scaler.scale(loss).backward()

                # debug step
#                 if model.module._check_nan_grads():
#                     print("NaN gradients detected, logging...")
#                     print(f"i={i}, minibatch={minibatch}, batch_path={batch_path}, global_step={global_step}")

                scaler.unscale_(optimizer)

                # calculate dynamic gradient clip threshold
                obs_grad_norm += float(model.module._get_grad_norm()) / step_len

                if not static_clip:
                    grad_history.append(obs_grad_norm)
                    grad_history = grad_history[-history_size:]
                    clip_value = np.mean(grad_history)  

                # clip gradients
                if clip_value != -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                # update weights
                scaler.step(optimizer)

                # if lr scheduler is used
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # update scaler's scale value and other parameters
                scaler.update()
                if scaler.get_scale() > 2.0**17:
                    scaler.update(2.0**18)

                total_loss += loss.item() / step_len
                minibatch += 1
                global_step += 1

                if minibatch % log_interval == 0:

                    cur_loss = min(total_loss / log_interval, 1e1)
                    obs_grad_norm /= log_interval
                    lr = optimizer.param_groups[0]['lr']
                    ppl = math.exp(cur_loss)
                    bpt = cur_loss / math.log(2)

                    if writer is not None and rank == 0:
                        writer.add_scalar('training/loss', cur_loss, global_step)
                        writer.add_scalar('training/gnorm', obs_grad_norm, global_step)
                        writer.add_scalar('training/scale', scaler.get_scale(), global_step)
                        writer.add_scalar('training/ppl', ppl, global_step)
                        writer.add_scalar('training/bpt', bpt, global_step)
                        writer.add_scalar('training/lr', lr, global_step)

                    print(f'| rank {rank} | global step {global_step:0>6} |  lr {lr:5.5f} | loss {cur_loss:5.2f} | gnorm {obs_grad_norm:8.2f} |')
                    total_loss = 0
                    obs_grad_norm = 0

                if global_step >= max_steps:
                    return global_step

    return global_step


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
    best_loss, global_step = 1e3, args.global_step
    if args.load_checkpoint:
        load_states_from_checkpoint(torch.load(args.load_checkpoint), model, optimizer, scaler, lr_scheduler)
        print(f"Loaded checkpoint model", args.load_checkpoint)
        
    print("No of parameters", sum(p.numel() for p in model.parameters()))
    model = DummyDDPWrapper(model)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = args.warmup,
                                                    num_training_steps = args.max_steps,
                                                    num_cycles = 0.35
                                                  )

    writer = SummaryWriter(args.writer_dir)

    for epoch in range(args.epochs):
        random.shuffle(batches)
        for batch_path in batches:
            # global step is a counter of update steps
            print("Processing", batch_path)
            global_step = train(
                model,
                batch_path,
                optimizer,
                scaler,
                global_step=global_step,
                eval_steps=args.eval_steps,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                clip_value=args.clip_value,
                seq_len=args.bptt,
                writer=writer,
                use_amp=use_amp,
                device=device,
                lr_scheduler=lr_scheduler,
                checkpoint_path=args.checkpoint_path
            )

            print(f"Step = {global_step} : finished batch {batch_path}")

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break
        
        print(f"Finished epoch {epoch}")

    print("Starting Final Evaluation")

    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }

    torch.save(checkpoint, args.checkpoint_path + f"_last.ckpt")
    evaluate_dir(model.module, args.val_dir, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, use_amp=use_amp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_lr", type=float, default=15e-4)
    parser.add_argument("--clip_value", type=float, default=0.5)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=100)

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str, default="bin/enwik8/base/base")
    parser.add_argument("--writer_dir", type=str, default="runs/enwik8-base")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--global_step", type=int, default=0)

    parser.add_argument("--model_config", type=str, default="config/small_exp.json")

    args = parser.parse_args()

    print()
    print(args)
    print()

    main(args)