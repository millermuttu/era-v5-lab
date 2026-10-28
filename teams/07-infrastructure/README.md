# Team 07-infrastructure

Put your work in a **topic folder**: `teams/07-infrastructure/<topic>/<file>`, for example `teams/07-infrastructure/harness/run.py`.
The team lead decides the topic names. Do not make personal folders. Do not make new folders under `teams/`.
Open a pull request; the lead reviews and merges it. The pr-guard check must pass.

## Work so far (week 1)

| Topic | What it is | Start here |
|---|---|---|
| `queue/slurm-docker/` | Slurm 25.11 cluster in Docker (controller, accounting, two compute nodes with 8 fake GPUs each) to test accounts, GPU shares and multi-node jobs without a GPU. The same `slurm.conf` is used on the real node; only `site.conf`, `gres.conf`, `cgroup.conf`, `slurmdbd.conf` and `accounts.cfg` differ (`docker/` vs `node/`). | [README](queue/slurm-docker/README.md), `./test.sh` |
| `health/` | Node checks around a job. `gpu_health.sh`: pre-flight check of GPU count, temperature, ECC, row remap and NVLink. `xid_scan.py`: flags hardware Xid errors in the kernel log (exit 1 = drain the node). `allreduce.py`: NCCL all-reduce that checks the sum and measures bus bandwidth (gloo on CPU). `launch.sh`: `sbatch` job that runs all three in order. | [TESTING](health/TESTING.md) |
| `checkpoint/` | Save and restore of the full training state with PyTorch DCP: sharded fp32 weights and optimizer, scheduler, RNG and data position. A checkpoint counts only once `COMPLETE` is written, so a crash during a save never breaks the previous one. Loads on any GPU count or on CPU. | [DESIGN](checkpoint/DESIGN.md), `python ckpt.py` |

Open item: the `gpu_health.sh` output format may change to match the Team 8 health-check JSON format (due 29 Oct).
