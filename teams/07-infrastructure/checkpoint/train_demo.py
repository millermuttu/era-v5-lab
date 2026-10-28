"""A small training run that restarts itself from the newest COMPLETE checkpoint. Started by restart.sh.

It is the shape the real training script must have: load() at the start, save() every SAVE_EVERY steps.
Every start, first or after a requeue, runs the same code; load() returns None on the first one.
One process per GPU under torchrun, FSDP2; with no GPU it runs on the CPU over gloo.
Knobs: ROOT (checkpoint folder, required), STEPS (default 10), SAVE_EVERY (default 1),
SLOW (seconds to sleep each step, default 0, so a test can kill it in the middle).
Rank 0 appends "step loss" lines to ROOT/losses.txt and, at the end, the sha256 of the full weights to ROOT/final.txt.
"""
import hashlib
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.fsdp import fully_shard

import ckpt

root = Path(os.environ["ROOT"])
root.mkdir(parents=True, exist_ok=True)
steps = int(os.environ.get("STEPS", 10))
every = int(os.environ.get("SAVE_EVERY", 1))
slow = float(os.environ.get("SLOW", 0))
meta = {"config": "train_demo", "tokenizer": "none"}

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl" if cuda else "gloo")
rank = dist.get_rank()
dev = "cuda" if cuda else "cpu"

torch.manual_seed(1234)  # the same initial weights on every rank
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
                            torch.nn.Linear(64, 1)).to(dev)
fully_shard(model)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
torch.manual_seed(1234 + rank)  # each rank its own dropout stream

start, extra = ckpt.load(root, model, opt, sched, meta) or (0, {})
if rank == 0:
    print(f"start at step {start}, restart {os.environ.get('SLURM_RESTART_COUNT', 0)}", flush=True)

for step in range(start, steps):
    g = torch.Generator().manual_seed(step * 1000 + rank)  # the batch is a function of the step and the rank
    x, y = torch.randn(8, 16, generator=g).to(dev), torch.randn(8, 1, generator=g).to(dev)
    loss = torch.nn.functional.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    dist.all_reduce(loss.detach())
    if rank == 0:
        with open(root / "losses.txt", "a") as f:
            f.write(f"{step} {loss.item()!r}\n")
    time.sleep(slow)
    if (step + 1) % every == 0:
        ckpt.save(root, step + 1, model, opt, sched, {"data": {"samples": (step + 1) * 8}}, meta=meta)

sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
if rank == 0:
    h = hashlib.sha256(b"".join(t.numpy().tobytes() for t in sd.values())).hexdigest()
    (root / "final.txt").write_text(h + "\n")
    print(f"done at step {steps}, weights {h[:16]}", flush=True)
dist.destroy_process_group()
