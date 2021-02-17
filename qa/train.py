import random
import argparse

import numpy as np
import torch.nn as nn
import torch

from tokenizer import SHALSTMTokenizer
from .model import SHALSTMforQuestionAnswering 
from .eval import chunks, load_dataset_with_ids, get_predictions

from apex.optimizers import FusedLAMB
from tensorboardX import SummaryWriter
from datasets import load_metric


def load_tsv_dataset(dir, tokenizer, batch_size=32):

    with open(dir) as fi:
        lines = fi.read().splitlines()
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


def validate_f1(tokenizer, model, val_data, use_amp=True):
    
    val_data, val_gold = val_data
    predictions = get_predictions(model, val_data, use_amp=use_amp)

    tokenizer.decode_batch(predictions)

    decoded = tokenizer.decode_batch(predictions)
    decoded = [ s.split("</s>")[0] for s in decoded ]

    metric = load_metric("squad")

    predictions = [{'prediction_text':x , "id": str(y) } for y, x in enumerate(decoded, start=1) ]
    references = [{'answers': {'answer_start': [1]*len(x), 'text': x  }, 'id': str(y) } for y, x in enumerate(val_gold, start=1) ]
    
    result = metric.compute(predictions=predictions, references=references)
    return result


def train(
        model,
        train_data,
        optimizer,
        scaler,
        lr_scheduler,
        global_step=0,
        log_interval=100,
        warmup=False,
        writer=None,
        clip_value=-1,
        static_clip=True,
        history_size=100, # this is used only if static_clip is False 
        use_amp=True
        ):

    total_loss, minibatch = 0, 0
    model.train()
    
    for input, attn_mask, type_ids, input_length in train_data:

        # zero out gradients
        model.zero_grad()
        
        # calculate loss
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, h, hidden, mems = model(input, attn_mask, type_ids, return_loss=True)

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
    
    train_data, train_gold = load_dataset_with_ids(args.train_dir, args.train_ans, tokenizer, batch_size=args.batch_size)
    val = load_dataset_with_ids(args.val_dir, args.val_ans, tokenizer, batch_size=args.batch_size)

    warmup = args.warmup * len(train_data)
    total_steps = args.epochs * len(train_data)

    if args.device == "cuda":
        optimizer = FusedLAMB(model.parameters(), lr=args.base_lr, max_grad_norm=args.clip_value, use_nvlamb=True)
    else:
        optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: float(x / warmup) if x < warmup else float((total_steps - x) / total_steps)])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(args.writer_dir)

    global_step = 0
    best_loss = 1e3
    best_f1 = 0

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
            log_interval=100,
            warmup=False,
            writer=writer,
            use_amp=use_amp
        )

        # val_loss = validate_loss(model, val_data, use_amp=True)
        # writer.add_scalar('validation/loss', val_loss, global_step)

        val_result = validate_f1(tokenizer, model, val, use_amp=use_amp)
        
        writer.add_scalar('validation/f1', val_result["f1"], global_step)
        writer.add_scalar('validation/exact_match', val_result["exact_match"], global_step)

        if valf1 > best_f1 and args.checkpoint_dir:
            print("saving checkpoint...")
            model.save(args.checkpoint_dir)

    model.load(args.checkpoint_dir + ".pt")
    model.to(device)

    val_result = validate_f1(tokenizer, model, val, use_amp=use_amp)

    writer.add_scalar('train/f1', val_result["f1"], global_step)
    writer.add_scalar('train/exact_match', val_result["exact_match"], global_step)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--clip_value", type=float, default=0.1)
    # parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--train_ans", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--val_ans", type=str, required=True)
    # parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--writer_dir", type=str, default="qa-runs/base-qa")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    main(args)