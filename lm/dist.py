import os
import sys
import traceback
import tempfile
import argparse
import datetime
import time
import math
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from shalstm.model import SHALSTM
from apex.optimizers import FusedLAMB
from shalstm.optim import MinTrustLamb
from .train import train, evaluate_dir, load_states_from_checkpoint

from tensorboardX import SummaryWriter
from glob import glob
from tqdm import trange
import numpy as np


def run_proc(local_rank, args):

    device_ids = list(range(local_rank, local_rank + 1))
    device = torch.device(device_ids[0])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Initialized process {local_rank}, with devices {device_ids}, global rank {rank}\n")

    no_batches = len(glob(args.train_dir + "*.pt"))
    args.evaluate_each = min(args.evaluate_each, no_batches // world_size)
    start = 0

    local_batches = [ f"{args.train_dir}batch_{((rank + i) % no_batches ) + 1:0>5}.pt" for i in range(start, args.epochs * world_size * math.ceil(no_batches / world_size), world_size) ]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = SHALSTM(args.model_config, device=device)
    global_step = 0

    model = DDP(model, device_ids)
    print(f"[{os.getpid()}] initialized ddp model.")

    CHECKPOINT_PATH = args.checkpoint_path + ".init.pt"

    if rank == 0:
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"[{os.getpid()}] Saved initialized model.")

    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    if args.optimizer == "fused":
        optimizer = FusedLAMB(model.parameters(), lr=args.base_lr, max_grad_norm=args.clip_value, use_nvlamb=True)
    else:
        optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)

    # start from checkpoint
    if args.load_checkpoint:
        load_states_from_checkpoint(torch.load(args.load_checkpoint), model, optimizer, scaler)
        global_step = optimizer.state[0]["step"]
        print(f"[{os.getpid()}] Loaded checkpoint model", args.load_checkpoint)
        print(f"[{os.getpid()}] Starting with global step =", global_step)

    if rank == 0:
        writer = SummaryWriter(args.writer_dir)
    else:
        writer = None

    print(f"rank = {rank}, no batches = {no_batches}")

    epoch = 1
    for i, batch_path in enumerate(local_batches):
        print(f"Loading {batch_path}")

        # global step is just a counter of batches
        global_step = train(
            model,
            batch_path,
            args.batch_size,
            optimizer,
            scaler,
            args.base_lr,
            world_size=world_size,
            rank=rank,
            global_step=global_step,
            log_interval=args.log_interval,
            clip_value=args.clip_value if args.optimizer != "fused" else -1,
            seq_len=args.bptt,
            warmup=args.warmup,
            writer=writer,
            device=device
        )


        if rank == 0:
            if i % args.evaluate_each == 0 and i > 0:
                # evaluate_dir(model.module, args.val_dir + "*.pt", args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
                pass

            model.module.save(args.checkpoint_path + f"_{global_step}")

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }

            torch.save(checkpoint, CHECKPOINT_PATH)

        dist.barrier()

        # resync model, optimizer, scaler on all gpus (just to make sure)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
        load_states_from_checkpoint(checkpoint, model, optimizer, scaler)
        print(f"finished {batch_path}")

        # if (i + 1) == (len(local_batches) // args.epochs):
        #     if rank == 0:
        #         print(f"Finished epoch {epoch}")
        #         epoch += 1
        #         evaluate_dir(model.module, args.val_dir + "*.pt", args.batch_size, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
        #         evaluate_dir(model.module, args.test_dir+ "*.pt", args.batch_size, set="test", writer=writer, global_step=global_step, seq_len=args.bptt, device=device)
        #         model.module.save(args.checkpoint_path + f"_{global_step}")

        #     dist.barrier()
        #     args.base_lr /= 2


def spmd_main(local_rank, args):

    try:
        process_rank = local_rank + args.rank
        dist.init_process_group(backend="nccl",
                                init_method="file://" + args.dist_file, 
                                rank=process_rank, 
                                world_size=args.world_size,
                                timeout=datetime.timedelta(0, 60 * 60 *2))

        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        run_proc(local_rank, args)

    except Exception:
        traceback.print_exc()

    # Tear down the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist_file", type=str, default="dist_shared")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--base_lr", type=float, default=2e-3)
    parser.add_argument("--optimizer", type=str, default="fused")
    parser.add_argument("--clip_value", type=float, default=0.1)
    parser.add_argument("--bptt", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--evaluate_each", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--train_dir", type=str, default="/userfiles/asafaya19/pile/train/")
    parser.add_argument("--val_dir", type=str, default="/userfiles/asafaya19/pile/val/")
    parser.add_argument("--test_dir", type=str, default="/userfiles/asafaya19/pile/test/")
    
    parser.add_argument("--checkpoint_path", type=str, default="bin/pile/small/small")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--model_config", type=str, default="config/small.json")
    parser.add_argument("--writer_dir", type=str, default="runs/small")

    args = parser.parse_args()

    spmd_main(args.local_rank, args)