# Checkpoint design

Team 7, week 1. Code: `ckpt.py`. Test: `python ckpt.py` (CPU, a few seconds).

## Goal

The training run (a 30B MoE model, about the shape of Qwen3-30B-A3B) must survive the loss of a rank or a node and lose no work.

1. A restart resumes from the newest COMPLETE checkpoint and gives bit-equal loss to a run that never stopped.
2. A crash during a save never breaks the previous checkpoint.
3. Any checkpoint can be read on a machine with no GPU (Team 9 evaluation, Team 12 quantization) and on a different number of GPUs.

## What a checkpoint holds

| Part | Why | Where |
|---|---|---|
| Model weights in fp32, sharded | Resume and evaluate | DCP shards |
| Optimizer state (AdamW `exp_avg`, `exp_avg_sq`, step) | Without it, the loss jumps after a restart | DCP shards |
| LR scheduler state | Same learning rate after a restart | `rank<N>.pt` |
| Step | Where to continue | `rank<N>.pt`, `COMPLETE` |
| RNG state of Python, the CPU and this rank's GPU | Same dropout and router noise after a restart | `rank<N>.pt` |
| Data position (see below) | Same batches after a restart | `rank<N>.pt`, in `extra` |

Only the fp32 copy is saved. The bf16 compute copy is made from it again, so saving it would only cost time.

## Format: PyTorch Distributed Checkpoint (DCP)

`torch.distributed.checkpoint` writes one file per rank, in parallel, and one `.metadata` index. It works with FSDP2, tensor parallel and expert parallel (DTensor), so the parallel layout chosen in week 3 does not change the format. At load time it reshards: a checkpoint saved on 64 GPUs loads on 32 GPUs or on a CPU.

```
<root>/step_00012000/
  .metadata          index: which piece of which tensor is in which file
  __0_0.distcp ...   tensor shards, one file per rank
  rank0.pt ...       per rank: step, RNG, scheduler, extra (data position)
  COMPLETE           written last; holds the step number
```

## Commit rule: COMPLETE is written last

1. All ranks write into `step_<N>.tmp/`.
2. All ranks wait at a barrier.
3. Rank 0 writes `COMPLETE`, then renames the folder to `step_<N>/`.

A restart, the evaluation watcher and the quantization flow only read a folder that has `COMPLETE` (`ckpt.latest(root)`). A save that crashes leaves a `.tmp` folder, which the next save deletes. S3 has no rename, so the same rule holds there by upload order: upload every other file first, `COMPLETE` last. A checkpoint is never found by listing a folder alone.

## Data position format

This is what Teams 1 and 2 asked for. Proposal:

```python
extra = {"data": {
    "manifest": "sha256 of the shard manifest list",
    "seed": 1234,
    "samples": 4_096_000,   # global samples consumed, all data-parallel ranks together
    "tokens": 16_777_216_000,
}}
```

The loader turns a global sample index into (shard, row) from the seed alone, and each data-parallel rank takes its slice of each global batch. The position is then one number, and it does not depend on the number of GPUs. On restore the loader checks `manifest`. A different shard list must stop the run, not resume silently on other data. The self-test uses the same idea: the batch is a function of the step.

## Size and time

Config v0 (Team 5) leaves the model sizes and the precision empty until TSAI fills them on 31 Oct, so the numbers below assume 30B parameters and fp32 master weights.

30B parameters × (4 bytes weights + 8 bytes Adam) = about 360 GB per checkpoint.

- **Save to local NVMe or shared disk.** Every rank writes in parallel. At 3 GB/s per node, 8 nodes save in about 15 s. Team 6 measures the real time at config v1 shape (week 2) and at full size (18 Nov).
- **Upload to S3 in the background** after the commit, off the training path. TSAI keeps the S3 keys. They reach the node through the environment or an instance role, never through the repository.
- **Keep** the newest 3 on local disk (`keep=3`), and every checkpoint whose step is a multiple of `keep_every` as a milestone for evaluation and for rollback after a loss spike. S3 keeps all milestones.

## Checkpoint interval

Set in week 3 from measured numbers with the Young/Daly rule:

```
interval = sqrt(2 × save_time × mean_time_between_failures)
```

Llama 3 had 419 unplanned stops in 54 days on 16,384 GPUs, about one stop per 50,000 GPU-hours. On 64 GPUs that is about one stop per month. With a 60 s save the rule gives about 5 h. We start at 1 h, because the dry runs need many restore points and a new cluster fails more often. Then we lengthen it from what dry run 1 measures.

## When the loss is bit-equal

A restore gives bit-equal loss only when all of these are the same as before the stop:

- number of GPUs and parallel layout (data, tensor and expert parallel sizes)
- GPU type (H100 and B200 run different kernels)
- PyTorch, CUDA and NCCL versions, and the kernel set (Team 6 freezes it on 23 Nov)
- data order (seed and manifest; config v0 sets `run.seed: 1234`)
- attention kernel (config v0 sets `attention.impl: eager`, because the fast attention backward does not repeat)

If one of them changes, the restore still works, but the loss is only close: a different reduction order changes the last bits. The restart scripts must keep the layout, and the runbook must say this. The planned failure in dry run 2 gets bit-equal loss only if the job comes back on the same layout.

## What the test proves

`python ckpt.py` trains a small model with dropout, AdamW and a step LR schedule on the CPU. It runs 10 steps straight, then 5 steps with a save after each step. It restores into a new model with other weights and another seed, and runs steps 5 to 9. All 10 losses must be bit-equal (`==` on floats). With the RNG restore turned off, the test fails, so the RNG state is needed. The test also checks that a `.tmp` folder and a folder with no `COMPLETE` are skipped, and that `keep` and `keep_every` prune the right folders. A model saved on the GPU and restored on the CPU had bit-equal weights and Adam state (checked by hand, 27 Oct).

## Config v0

Team 5 config v0 (`shivman/configs/config_v0.yaml` in `udisinghania/ERA_V5_Architecture`) needs no change to `ckpt.py`:

- `optim.name: adamw`: the Adam state is already saved.
- `run.seed: 1234`: the same seed as the data position proposal above.
- `loss.train_on: ntp`, with a second head trained on t+2: that head is part of the model state dict, so it is saved with the rest.
- `embedding.type: kronecker`: its parameters are in the model state dict too.
- `run.grad_clip: 1.0`: the harness clips before the optimizer step; nothing to save.

## Reading a checkpoint outside training

Teams 9 and 12 make one plain file on a CPU machine:

```python
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save(ckpt.latest(root), "model.pt")   # then keep only the "model" part
```

## Not built yet

- **S3 upload and download.** Week 2, once TSAI gives the bucket.
- **Automatic restart** (12 to 14 Nov). `sbatch --requeue`, and the training script calls `load()` at start.
- **Async save** (`dcp.async_save`). Only if the measured save time is a real share of the interval.
- **fsync of `rank<N>.pt`.** A power cut right after the commit could leave it empty. The previous checkpoint is still there.

## Open questions

1. Local NVMe or a shared file system on the training nodes? This sets the save time.
2. Do Teams 1 and 2 accept the data position format?
3. Config v0 leaves `precision` empty. Does config v1 (Team 5, 14 Nov) keep master weights in fp32? If it keeps them in bf16, the checkpoint is about 240 GB.
4. The plan sheet says H100 and B200. Do we also need A100 nodes?
