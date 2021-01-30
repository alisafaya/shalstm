import os
import sys
import traceback
import tempfile
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from shalstm.model import SHALSTM
from shalstm.optim import MinTrustLamb
from tqdm import trange

import time
import math

def demo_basic(local_world_size, local_rank):

    n = 1
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    device = torch.device(device_ids[0])
    rank = int(os.environ["RANK"])
    
    print(f"Initialized process {local_rank}, with devices {device_ids}, rank {rank}\n")

    use_amp = True
    model = SHALSTM("config/small.json", device=device)
    model = DDP(model, device_ids)

    CHECKPOINT_PATH = "init_model.checkpoint"

    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Saved initialized model")

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = MinTrustLamb(model.parameters(), lr=1e-3)
    train_data = torch.load(f"/userfiles/asafaya19/pile/train/batch_{(rank + 1):0>5}.pt").long() # input size x batch size
    from .train import batchify, get_batch
    train_data = batchify(train_data, 16)

    hidden, mems = None, None
    # range_func = trange if local_rank == 0 else range
    range_func = range

    starttime = time.time()
    i = 0
    batch = 0
    seq_len = 1024
    total_loss = 0
    
    while i < train_data.size(0) - 1 - 1:
    
        data, targets = get_batch(train_data, i, seq_len)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, h, hidden, mems = model(data, targets=targets, hidden=hidden, mems=mems)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        if rank == 0:
            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / 100
                elapsed = time.time() - starttime
                print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f} | bpt {:8.3f}'.format(batch, len(train_data) // seq_len, elapsed * 1000 / 100, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                total_loss = 0
            batch +=1 

        i += seq_len

    if local_rank == 0:
        print("Excecution time =", (time.time() - starttime) / len(train_data), "sec per batch")
        print("Total excecution time =", (time.time() - starttime), "sec per batch")


def spmd_main(local_world_size, local_rank):
    
    try:
        dist.init_process_group(backend="nccl")
        
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        demo_basic(local_world_size, local_rank)

    except Exception:
        if local_rank == 0:
            traceback.print_exc() 

    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()
    spmd_main(args.local_world_size, args.local_rank)


