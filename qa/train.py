import random
import argparse
from glob import glob

import numpy as np
import torch.nn as nn
import torch

from tokenizer import SHALSTMTokenizer
from .model import SHALSTMforQuestionAnswering 
from .eval import chunks, load_dataset_with_ids, get_predictions

from apex.optimizers import FusedLAMB
from tensorboardX import SummaryWriter
from datasets import load_metric

def load_states_from_checkpoint(checkpoint, model=None, optimizer=None, scaler=None):
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

def load_tsv_dataset(dir, tokenizer, batch_size=32):

    lines = []
    for f in glob(dir):
        with open(f) as fi:
            lines += fi.read().splitlines()

    random.shuffle(lines)

    questions, answers = tuple(np.array(x) for x in zip(*[ l.split("\t") for l in lines ]))
    q_lens = np.array([ len(tokenizer.encode(x)) for x in questions ])
    indices = np.argsort(q_lens)

    questions = list(chunks(list(questions[indices]), batch_size))
    answers = list(chunks(list(answers[indices]), batch_size))

    batches = []
    for q, a in zip(questions, answers):
        batches.append(tokenizer.encode_for_qa(q, a))

    return batches


def validate_loss(
        model,
        val_data,
        use_amp=True
        ):

    total_loss, minibatch = 0, 0
    model.eval()
    with torch.no_grad():
        for input, attn_mask, type_ids, input_length in val_data:
            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, h, hidden, mems = model(input, attn_mask, type_ids, return_loss=True)

            total_loss += loss.item()
            minibatch += 1

    total_loss /= minibatch

    return total_loss


def train(
        model,
        train_data,
        optimizer,
        scaler,
        lr_scheduler,
        global_step=0,
        log_interval=100,
        writer=None,
        clip_value=-1,
        static_clip=True,
        history_size=100, # this is used only if static_clip is False 
        use_amp=True,
        use_lm_loss=False
        ):

    total_loss, minibatch = 0, 0
    model.train()
    
    for input, attn_mask, type_ids, input_length in train_data:

        # zero out gradients
        model.zero_grad()
        
        # calculate loss
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, h, hidden, mems = model(input, attn_mask, type_ids, return_loss=True, lm_loss=use_lm_loss)

        # do backward pass
        scaler.scale(loss).backward()
        
        # debug step
        if model._check_nan_grads():
            print("NaN gradients detected, logging...")
            print(f"minibatch={minibatch}, global_step={global_step}")

        scaler.unscale_(optimizer)

        # calculate dynamic gradient clip threshold
        if not static_clip:
            obs_grad_norm = model._get_grad_norm()
            grad_history.append(obs_grad_norm)
            grad_history = grad_history[-history_size:]
            clip_value = np.mean(grad_history)  

        # clip gradients
        if clip_value != -1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # update weights
        if not model._check_nan_grads():
            scaler.step(optimizer)
        
        # update scaler's scale value and other parameters
        scaler.update()

        # update lr
        lr_scheduler.step()

        total_loss += loss.item()
        if minibatch % log_interval == 0 and minibatch > 0:

            if writer is not None:
                writer.add_scalar('training/loss', min(total_loss / log_interval, 1e1), global_step)
                writer.add_scalar('training/scale', scaler.get_scale(), global_step)
                writer.add_scalar('training/lr', optimizer.param_groups[0]['lr'], global_step)

            total_loss = 0

        minibatch += 1
        global_step += 1

    return global_step


def main(args):

    device = torch.device(args.device)
    use_amp = args.no_amp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model = SHALSTMforQuestionAnswering.from_pretrained(args.model, device=torch.device(args.device))
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)

    train_data = load_tsv_dataset(args.train_dir, tokenizer, batch_size=args.batch_size)
    val_data = load_tsv_dataset(args.val_dir, tokenizer, batch_size=args.batch_size)
    
    warmup = args.warmup * len(train_data)
    total_steps = args.epochs * len(train_data)

    if args.device == "cuda":
        optimizer = FusedLAMB(model.parameters(), lr=args.base_lr, max_grad_norm=args.clip_value, use_nvlamb=True)
    else:
        optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: float(x / warmup) if x < warmup else float((total_steps - x) / total_steps)]) # warmup with linear decay

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: float(x / warmup) if x < warmup else float(1.)]) # warmup with linear decay # warmup with no decay

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    if args.writer_dir:
        writer = SummaryWriter(args.writer_dir)
    else:
        writer = None

    global_step = 0
    best_loss = 1e3
    best_f1 = 0

    if args.load_checkpoint:
        load_states_from_checkpoint(torch.load(args.load_checkpoint), model=model, optimizer=optimizer, scaler=scaler)

    for epoch in range(1, args.epochs+1):

        print("Epoch", epoch)
        random.shuffle(train_data)
        
        global_step = train(
            model,
            train_data,
            optimizer,
            scaler,
            lr_scheduler,
            global_step=global_step,
            log_interval=args.log_interval,
            writer=writer,
            use_amp=use_amp,
            use_lm_loss=args.pretraining
        )

        val_loss = validate_loss(model, val_data, use_amp=use_amp)
        writer.add_scalar('validation/loss', val_loss, global_step)

        if args.checkpoint_dir:
            print("saving checkpoint...")
            model.save(args.checkpoint_dir + "_" + str(epoch))

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }
#             torch.save(checkpoint, args.checkpoint_dir + "_" + str(epoch) + ".ckpt")

            best_loss = val_loss


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pretraining", action="store_true")

    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--clip_value", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)

    parser.add_argument("--writer_dir", type=str, default="")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    main(args)