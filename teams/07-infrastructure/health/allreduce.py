"""NCCL all-reduce test: checks the sum on every rank and measures bus bandwidth.

Started by torchrun (see launch.sh), one process per GPU.
Knobs: SIZE_MB (default 1024), ITERS (default 20), MIN_BUSBW_GBPS (default 0, no limit).
Exits 1 if the sum is wrong on any rank or the bandwidth is below MIN_BUSBW_GBPS.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

size_mb = int(os.environ.get("SIZE_MB", 1024))
iters = int(os.environ.get("ITERS", 20))
min_busbw = float(os.environ.get("MIN_BUSBW_GBPS", 0))

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
x = torch.empty(size_mb * 2**20 // 4, device="cuda")

# Correctness: every rank adds its ones, so each element must equal the world size.
x.fill_(1.0)
dist.all_reduce(x)
ok = torch.tensor(int(torch.all(x == world).item()), device="cuda")
dist.all_reduce(ok, op=dist.ReduceOp.MIN)

# Bandwidth, on zeros so the sum never overflows.
x.zero_()
for _ in range(5):
    dist.all_reduce(x)
torch.cuda.synchronize()
dist.barrier()
t = time.perf_counter()
for _ in range(iters):
    dist.all_reduce(x)
torch.cuda.synchronize()
sec = (time.perf_counter() - t) / iters

# Same formula as nccl-tests: ring all-reduce moves 2(n-1)/n of the buffer over each link.
algbw = x.numel() * 4 / sec / 1e9
busbw = algbw * 2 * (world - 1) / world
passed = bool(ok.item()) and busbw >= min_busbw
if rank == 0:
    print(f"all-reduce {size_mb} MB on {world} GPUs: {sec * 1e3:.2f} ms, "
          f"algbw {algbw:.1f} GB/s, busbw {busbw:.1f} GB/s, "
          f"sum {'correct' if ok.item() else 'WRONG'} -> {'PASS' if passed else 'FAIL'}")
dist.destroy_process_group()
sys.exit(0 if passed else 1)
