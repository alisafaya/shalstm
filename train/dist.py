import os
import sys
import traceback
import tempfile
import argparse
import datetime
import time
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from shalstm.model import SHALSTM
from shalstm.optim import MinTrustLamb
from .train import train, evaluate_dir

from tensorboardX import SummaryWriter
from glob import glob
from tqdm import trange
import numpy as np

def run_proc(local_world_size, local_rank, args):

    device_ids = list(range(local_rank, local_rank + 1))
    device = torch.device(device_ids[0])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Initialized process {local_rank}, with devices {device_ids}, global rank {rank}\n")

    no_batches = len(glob(args.train_dir + "*.pt"))
    # no_batches = 200
    args.evaluate_each = min(args.evaluate_each, no_batches // world_size)

    local_batches = [ f"{args.train_dir}batch_{((rank + i) % no_batches ) + 1:0>5}.pt" for i in range(0, args.epochs * world_size * math.ceil(no_batches / world_size), world_size) ]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = SHALSTM(args.model_config, device=device)
    model = DDP(model, device_ids)

    CHECKPOINT_PATH = args.checkpoint_path + ".init.pt"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Saved initialized model.")

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()

    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)

    if rank == 0:
        writer = SummaryWriter(args.writer_dir)
    else:
        writer = None

    print(f"rank = {rank}, no batches = {no_batches}")

    global_step = 0
    epoch = 1
    for i, batch_path in enumerate(local_batches):

        # global step is just a counter of batches
        global_step = train(
            model,
            batch_path,
            args.batch_size,
            optimizer,
            scaler,
            args.base_lr,
            rank=rank,
            global_step=global_step,
            log_interval=args.log_interval,
            clip_value=args.clip_value,
            seq_len=args.bptt,
            warmup=args.warmup,
            writer=writer,
            device=device
        )

        if i % args.evaluate_each == 0 and i > 0:

            if rank == 0:
                evaluate_dir(model.module, args.val_dir + "*.pt", args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
                model.module.save(args.checkpoint_path + f"_{global_step}")
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            dist.barrier()

            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

        print(f"finished {batch_path}")

        if (i + 1) == (len(local_batches) // args.epochs):
            if rank == 0:
                print(f"Finished epoch {epoch}")
                epoch += 1
                evaluate_dir(model.module, args.val_dir + "*.pt", args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
                evaluate_dir(model.module, args.test_dir+ "*.pt", args.batch_size, set="test", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
                model.module.save(args.checkpoint_path + f"_{global_step}")

            dist.barrier()
            args.base_lr /= 2


def spmd_main(env_dict, local_world_size, local_rank, args):

    try:
        dist.init_process_group(backend="nccl",
                                init_method="file:///kuacc/users/asafaya19/shalstm/dist_shared", 
                                rank=int(env_dict["RANK"]), 
                                world_size=int(env_dict["WORLD_SIZE"]),
                                timeout=datetime.timedelta(0, 60 * 60 *2))
        
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        run_proc(local_world_size, local_rank, args)

    except Exception:
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
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--base_lr", type=float, default=3e-3)
    parser.add_argument("--clip_value", type=float, default=0.25)
    parser.add_argument("--bptt", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--evaluate_each", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument("--train_dir", type=str, default="/userfiles/asafaya19/pile/train/")
    parser.add_argument("--val_dir", type=str, default="/userfiles/asafaya19/pile/val/")
    parser.add_argument("--test_dir", type=str, default="/userfiles/asafaya19/pile/test/")
    
    parser.add_argument("--checkpoint_path", type=str, default="bin/pile/small/small")
    parser.add_argument("--model_config", type=str, default="config/small.json")
    parser.add_argument("--writer_dir", type=str, default="runs/small-5")

    args = parser.parse_args()
    spmd_main(env_dict, args.local_world_size, args.local_rank, args)


