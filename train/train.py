import torch
import time
import math
import random
import numpy as np
from glob import glob

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
        data = source[-(seq_len+1): -1]
        target = source[-seq_len:]
    else:
        data = source[step:step+seq_len]
        target = source[step + 1:step + 1 + seq_len]
    
    return data, target

def warmup_lr(optimizer, lr, ratio):
    for param_group in optimizer.param_groups:        
        param_group['lr'] = lr * ratio


def evaluate_dir(model, batch_dir, batch_size, set="val", writer=None, global_step=0, seq_len=1024, device=torch.device("cuda")):
    
    print("Evaluating", batch_dir)

    loss, total_len = 0, 0
    for batch in glob(batch_dir):
        l, t = evaluate(model, batch, batch_size, seq_len=seq_len, device=device)
        loss += l
        total_len += t

    loss = loss / total_len
    ppl = math.exp(loss)
    bpt = loss / math.log(2)

    if writer is not None:
        writer.add_scalar(f'{set}/loss', loss, global_step)
        writer.add_scalar(f'{set}/ppl', ppl, global_step)
        writer.add_scalar(f'{set}/bpt', bpt, global_step)

    print("Finished evaluation")

    return loss

def evaluate(model, batch_path, batch_size, seq_len=1024, device=torch.device("cuda")):

    data = torch.load(batch_path).long()
    data = batchify(data, batch_size)#.to(device)
    # data = batchify(data, batch_size)[:1024 * 32].to(device)

    total_loss, i = 0, 0
    hidden, mems = None, None
    model.eval()
    with torch.no_grad():
        while i < data.size(0) - 2:
            # get minibatch
            data, targets = get_batch(data, i, seq_len=seq_len)
            
            # calculate loss
            loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)
            total_loss += loss.item() * len(data)
            i += seq_len

    return total_loss, i

def train(
        model,
        batch_path,
        batch_size,
        optimizer,
        scaler,
        base_lr,
        rank=0,
        global_step=0,
        seq_len=1024,
        log_interval=100,
        warmup=False,
        writer=None,
        clip_value=0.25,
        static_clip=True,
        history_size=100, # this is used only if static_clip is False 
        use_amp=True,
        device=torch.device("cuda")
        ):
    
    train_data = torch.load(batch_path).long()
    train_data = batchify(train_data, batch_size) #.to(device)
    # train_data = batchify(train_data, batch_size)[:1024 * (32 + random.randint(1, 10))] #.to(device)

    losses, grad_history = [], []

    total_loss, minibatch, i = 0, 0, 0
    hidden, mems = None, None
    model.train()
    
    with model.join():
        while i < train_data.size(0) - 2:

            # warm up
            ratio = (1 + min(global_step, warmup)) / warmup
            if ratio <= 1:
                warmup_lr(optimizer, base_lr, ratio)
            
            # get minibatch
            data, targets = get_batch(train_data, i, seq_len=seq_len)
            
            # zero out gradients
            optimizer.zero_grad()

            # calculate loss
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, output, hidden, mems = model(data, hidden=hidden, mems=mems, targets=targets)

            # do backward pass
            scaler.scale(loss).backward()
            
            # debug step
            if model.module._check_grads():

                print("NaN gradients detected, logging...")
                print(f"i={i}, minibatch={minibatch}, batch_path={batch_path}, global_step={global_step}")
                
                if rank == 0:
                    grad_dict = {k:v.grad for k, v in zip(model.module.state_dict(), model.module.parameters())}
                    for k in grad_dict:
                        if (grad_dict[k].data != grad_dict[k].data).any():
                            print("NaN grad at", k)
                        else:
                            print("No NaN grad at", k)

                    batch_id = batch_path.split("/")[-1]
                    torch.save(model.state_dict(), f"error_log/param_{global_step}_{batch_id}")
                    torch.save(grad_dict, f"error_log/grad_{global_step}_{batch_id}")
                    torch.save(optimizer.state_dict(), f"error_log/optim_{global_step}_{batch_id}")

                # minibatch += 1
                # i += seq_len
                # global_step += 1

                import sys; sys.exit("NaN gradients detected, terminating..")
            
            scaler.unscale_(optimizer)
            
            # calculate dynamic gradient clip threshold
            if not static_clip:
                obs_grad_norm = model.module._get_grad_norm()
                grad_history.append(obs_grad_norm)
                grad_history = grad_history[-history_size:]
                clip_value = np.mean(grad_history)  

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # update weights
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if minibatch % log_interval == 0 and minibatch > 0:

                cur_loss = min(total_loss / log_interval, 2e1)
                lr = optimizer.param_groups[0]['lr']
                ppl = math.exp(cur_loss)
                bpt = cur_loss / math.log(2)

                if writer is not None:
                    writer.add_scalar('training/loss', cur_loss, global_step)
                    writer.add_scalar('training/ppl', ppl, global_step)
                    writer.add_scalar('training/bpt', bpt, global_step)
                    writer.add_scalar('training/lr', lr, global_step)

                total_loss = 0

            minibatch += 1
            i += seq_len
            global_step += 1
    
    return global_step