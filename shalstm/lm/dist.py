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

from ..model import SHALSTM
from ..optim import MinTrustLamb
from .train import train, evaluate_dir, load_states_from_checkpoint
from transformers import get_cosine_schedule_with_warmup

from tensorboardX import SummaryWriter
from glob import glob
from tqdm import trange
import numpy as np


def run_proc(local_rank, args):

    device_ids = list(range(local_rank, local_rank + 1))
    device = torch.device(device_ids[0])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    use_amp = args.no_amp

    print(f"Initialized process {local_rank}, with devices {device_ids}, global rank {rank}\n")

    all_batches = sorted(glob(args.train_dir))
    no_batches = int(np.ceil(len(all_batches) / world_size))

    if rank == (world_size - 1):
        local_batches = all_batches[local_rank * no_batches:]
        local_batches += all_batches[:no_batches - len(local_batches)]
    else:    
        local_batches = all_batches[local_rank * no_batches: (local_rank + 1) * no_batches]

    print(f"process {local_rank}\n", local_batches)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = SHALSTM(args.model_config, device=device)
    global_step = 0

    model = DDP(model, device_ids, find_unused_parameters=True)
    print(f"[{os.getpid()}] initialized ddp model.")
    CHECKPOINT_PATH = args.checkpoint_path + ".init.pt"

    if rank == 0:
        print(args)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("No of parameters", sum(p.numel() for p in model.parameters()))
        print(f"[{os.getpid()}] Saved initialized model.")

    dist.barrier()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = MinTrustLamb(model.parameters(), lr=args.base_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = args.warmup,
                                                    num_training_steps = args.max_steps,
                                                    num_cycles = 0.25
                                                )
    # start from checkpoint
    if args.load_checkpoint:
        load_states_from_checkpoint(torch.load(args.load_checkpoint, map_location=map_location), model, optimizer, scaler, lr_scheduler)
        global_step = next(iter(optimizer.state.values()))["step"]
        print(f"[{os.getpid()}] Loaded checkpoint model", args.load_checkpoint)
        print(f"[{os.getpid()}] Starting with global step =", global_step)
        dist.barrier()

    if rank == 0:
        writer = SummaryWriter(args.writer_dir)
    else:
        writer = None

    print(f"rank = {local_rank}, no batches = {no_batches}")
    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch = {epoch}")
        for i, batch_path in enumerate(local_batches):
            print(f"Loading {batch_path}")

            # global step is just a counter of batches
            global_step = train(
                model,
                batch_path,
                optimizer,
                scaler,
                val_dir=args.val_dir,
                global_step=global_step,
                eval_steps=args.evaluate_each,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                clip_value=args.clip_value,
                seq_len=args.bptt,
                writer=writer,
                use_amp=use_amp,
                device=device,
                lr_scheduler=lr_scheduler,
                checkpoint_path=args.checkpoint_path,
                rank=rank,
                dist=dist
            )

            if local_rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': lr_scheduler.state_dict()
                }

                model.module.save(args.checkpoint_path + f"_{global_step}")
                torch.save(checkpoint, args.checkpoint_path + f"_last.ckpt")
                evaluate_dir(model.module, args.val_dir, set="val", writer=writer, global_step=global_step, seq_len=args.bptt, use_amp=use_amp)

            # resync model, optimizer, scaler on all gpus (just to make sure)
            print(f"Finished {batch_path}")
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(args.checkpoint_path + f"_last.ckpt", map_location=map_location)
            load_states_from_checkpoint(checkpoint, model, optimizer, scaler, lr_scheduler)

            if global_step >= args.max_steps:
                break


def spmd_main(local_rank, args):

    try:
        process_rank = local_rank + args.rank
        dist.init_process_group(backend="nccl",
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
    parser.add_argument("--clip_value", type=float, default=0.1)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--evaluate_each", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--no_amp", action="store_false")

    parser.add_argument("--train_dir", type=str, default="/userfiles/asafaya19/pile/train/")
    parser.add_argument("--val_dir", type=str, default="/userfiles/asafaya19/pile/val/")
    parser.add_argument("--test_dir", type=str, default="/userfiles/asafaya19/pile/test/")
    
    parser.add_argument("--checkpoint_path", type=str, default="bin/pile/small/small")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--model_config", type=str, default="config/small.json")
    parser.add_argument("--writer_dir", type=str, default="runs/small")

    args = parser.parse_args()

    spmd_main(args.local_rank, args)