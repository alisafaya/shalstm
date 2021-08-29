import random
import json
import argparse
from glob import glob
from functools import partial
from tqdm import trange, tqdm

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from ..tokenizer import SHALSTMTokenizer
from .model import SHALSTMforSequenceClassification
from ..optim import MinTrustLamb 
from apex.optimizers import FusedLAMB

from tensorboardX import SummaryWriter
from datasets import load_dataset, load_metric


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}


def process_data(tokenizer, inputs, labels, batch_size):

    input_ids, attn_mask = tokenizer.encode_as_tensors(inputs)
    labels = torch.tensor(labels)

    N = int(np.ceil(input_ids.shape[1] / batch_size))

    return list(zip(torch.chunk(input_ids, N, dim=1), torch.chunk(attn_mask, N, dim=1), torch.chunk(labels, N, dim=0)))


def get_metric(metric_, refs, preds):

    if metric_.config_name != "stsb":
        preds = np.argmax(preds, axis=1)
    else:
        preds = preds.flatten()
        refs = refs.flatten()

    result = metric_.compute(predictions=preds, references=refs)

    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    
    return result


def prepare_dataset(actual_task, tokenizer, batch_size=32):
    
    dataset = load_dataset("glue", actual_task)
    validation_key = "validation_matched" if actual_task == "mnli" else "validation"

    train, valid = dataset["train"], dataset[validation_key]
    sentence1_key, sentence2_key = task_to_keys[actual_task]

    is_regression = actual_task == "stsb"
    if not is_regression:
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    if sentence2_key is not None:
        train_data = process_data(tokenizer, list(zip(train[sentence1_key], train[sentence2_key])), train['label'], batch_size)
        valid_data = process_data(tokenizer, list(zip(valid[sentence1_key], valid[sentence2_key])), valid['label'], batch_size)
    else:
        train_data = process_data(tokenizer, train[sentence1_key], train['label'], batch_size)
        valid_data = process_data(tokenizer, valid[sentence1_key], valid['label'], batch_size) 

    metric = partial(get_metric, load_metric("glue", actual_task))

    return train_data, valid_data, num_labels, metric


def load_states_from_checkpoint(checkpoint, model=None, optimizer=None, scaler=None):
    if model is not None and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])


def validate(
        model,
        val_data,
        loss_fn=F.cross_entropy,
        use_amp=True
        ):

    model.eval()
    total_loss, minibatch = 0, 0
    references, predictions = torch.tensor([]), torch.tensor([])
    with torch.no_grad():
        for input, attn_mask, labels in val_data:
            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                output, h, hidden, mems = model(input, attn_mask)

                if loss_fn != F.cross_entropy:
                    output = output.view(-1)

                loss = loss_fn(output, labels.to(output.device))
                
            predictions = torch.cat([predictions, output.cpu()])
            references = torch.cat([references, labels.cpu()])

            total_loss += loss.item()
            minibatch += 1

    total_loss /= minibatch

    return total_loss, (references.numpy(), predictions.numpy())


def train(
        model,
        train_data,
        optimizer,
        scaler,
        lr_scheduler,
        task,
        loss_fn=F.cross_entropy,
        log_interval=100,
        writer=None,
        clip_value=-1,
        use_amp=True,
        global_step=0
        ):

    total_loss, minibatch = 0, 0
    model.train()

    for input, attn_mask, labels in tqdm(train_data, desc=f'Training for {task}'):

        # zero out gradients
        model.zero_grad()
        
        # calculate loss
        with torch.cuda.amp.autocast(enabled=use_amp):
            output, h, hidden, mems = model(input, attn_mask)
            
            if loss_fn != F.cross_entropy:
                output = output.view(-1)
                    
            loss = loss_fn(output, labels.to(output.device))

        # do backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

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
                writer.add_scalar(f'{task}/training/loss', min(total_loss / log_interval, 1e1), global_step)
                writer.add_scalar(f'{task}/training/scale', scaler.get_scale(), global_step)
                writer.add_scalar(f'{task}/training/lr', optimizer.param_groups[0]['lr'], global_step)
                
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

    # load dataset and prepare it
    tokenizer = SHALSTMTokenizer.from_file(args.tokenizer)
    train_data, val_data, num_labels, metric = prepare_dataset(args.task, tokenizer, batch_size=args.batch_size)

    # load model and tokenizer
    config = json.loads(open(args.model + ".json").read())
    model = SHALSTMforSequenceClassification(num_labels, config, device=device)
    model.load(args.model + ".pt")
    model.to(device)

    # set training configs
    total_steps = args.epochs * len(train_data)

    optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                        num_warmup_steps = len(train_data),
                                        num_training_steps = total_steps,
                                        num_cycles = 0.4)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(args.writer_dir) if args.writer_dir else None
    loss_fn = F.cross_entropy if args.task != 'stsb' else F.mse_loss

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
            args.task,
            loss_fn=loss_fn,
            log_interval=args.log_interval,
            writer=writer,
            clip_value=args.clip_value,
            use_amp=use_amp,
            global_step=global_step
        )

        val_loss, (refs, preds) = validate(model, val_data, loss_fn=loss_fn, use_amp=use_amp)
        eval_results = metric(refs, preds)

        if writer is not None:
            writer.add_scalar(f'{args.task}/validation/loss', val_loss, global_step)
            writer.add_text(args.task, str(eval_results), global_step)
        
        print("Evaluation of", args.task, "..")
        print(eval_results)

        if args.checkpoint_dir:

            print("saving checkpoint...")
            model.save(args.checkpoint_dir + "_" + str(epoch))

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }

            torch.save(checkpoint, args.checkpoint_dir + "_" + str(epoch) + ".ckpt")
            best_loss = val_loss


    val_loss, (refs, preds) = validate(model, val_data, loss_fn=loss_fn, use_amp=use_amp)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--clip_value", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--writer_dir", type=str, default="")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    main(args)