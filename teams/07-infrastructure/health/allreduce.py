"""NCCL collective test: checks the result on every rank and measures bus bandwidth.

OP picks the collective the training run depends on (Sessions 12 and 14):
  all_reduce      data-parallel gradients
  reduce_scatter  ZeRO / FSDP gradients
  all_gather      ZeRO-3 / FSDP weights
  all_to_all      MoE expert dispatch and combine
Started by torchrun (see launch.sh), one process per GPU. With no GPU it runs on the CPU over gloo,
for the Docker Slurm test; its bandwidth then means nothing.
Knobs: OP (default all_reduce), SIZE_MB (default 1024), ITERS (default 20), MIN_BUSBW_GBPS (default 0, no limit).
Exits 1 if the result is wrong on any rank or the bandwidth is below MIN_BUSBW_GBPS.
"""
import os
import sys
import time

import torch
import torch.distributed as dist

op = os.environ.get("OP", "all_reduce")
size_mb = int(os.environ.get("SIZE_MB", 1024))
iters = int(os.environ.get("ITERS", 20))
min_busbw = float(os.environ.get("MIN_BUSBW_GBPS", 0))

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl" if cuda else "gloo")
sync = torch.cuda.synchronize if cuda else (lambda: None)
rank, world = dist.get_rank(), dist.get_world_size()
dev = "cuda" if cuda else "cpu"
n = size_mb * 2**20 // 4 // world * world
x = torch.empty(n, device=dev)                # full buffer
part = torch.empty(n // world, device=dev)    # this rank's slice
ranks = torch.arange(world, device=dev)[:, None]

# Correctness. Sums: every rank adds its ones, so each element must equal the world size.
# Gather and all-to-all: every rank sends its own rank number, so slice i must hold i.
# Bus bandwidth factors are those of nccl-tests: the share of the buffer that crosses each link.
if op == "all_reduce":
    run, factor = (lambda: dist.all_reduce(x)), 2 * (world - 1) / world
    x.fill_(1.0); run(); good = torch.all(x == world)
elif op == "reduce_scatter":
    run, factor = (lambda: dist.reduce_scatter_tensor(part, x)), (world - 1) / world
    x.fill_(1.0); run(); good = torch.all(part == world)
elif op == "all_gather":
    run, factor = (lambda: dist.all_gather_into_tensor(x, part)), (world - 1) / world
    part.fill_(rank); run(); good = torch.all(x.view(world, -1) == ranks)
elif op == "all_to_all":
    y = torch.empty_like(x)
    run, factor = (lambda: dist.all_to_all_single(y, x)), (world - 1) / world
    x.fill_(rank); run(); good = torch.all(y.view(world, -1) == ranks)
else:
    sys.exit(f"unknown OP {op}: all_reduce, reduce_scatter, all_gather or all_to_all")
ok = torch.tensor(int(good.item()), device=dev)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)

# Bandwidth, on zeros so a sum never overflows.
x.zero_()
part.zero_()
for _ in range(5):
    run()
sync()
dist.barrier()
t = time.perf_counter()
for _ in range(iters):
    run()
sync()
sec = (time.perf_counter() - t) / iters

algbw = n * 4 / sec / 1e9
busbw = algbw * factor
passed = bool(ok.item()) and busbw >= min_busbw
if rank == 0:
    print(f"{op} {size_mb} MB on {world} GPUs: {sec * 1e3:.2f} ms, "
          f"algbw {algbw:.1f} GB/s, busbw {busbw:.1f} GB/s, "
          f"result {'correct' if ok.item() else 'WRONG'} -> {'PASS' if passed else 'FAIL'}")
dist.destroy_process_group()
sys.exit(0 if passed else 1)
