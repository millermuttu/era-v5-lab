# Slurm in Docker: queue and health test

Week 1 test for the job queue (Team 7). It needs no GPU: each compute node has 8 fake GPUs that Slurm counts but does not bind to a device, and a fake `nvidia-smi` that the health scripts read.

## Cluster

| Container | Role |
|---|---|
| `mariadb` | Accounting database |
| `slurmdbd` | Accounting daemon: accounts, users, limits |
| `slurmctld` | Controller |
| `c1`, `c2` | Compute nodes, 4 CPUs and 8 fake GPUs each |

Slurm 25.11 from Ubuntu 26.04. The Slurm 23.11 package in Ubuntu 24.04 cannot launch job steps in Docker, because it has no `CgroupPlugin=disabled`.

Every Slurm container mounts `teams/07-infrastructure/health` and `teams/07-infrastructure/checkpoint` read-only at `/lab/health` and `/lab/checkpoint`, the same path on every node, as a shared file system would. `/shared` is a writable volume on every node for job output.

## Files: one config for Docker and the real node

| File | Docker | Real node |
|---|---|---|
| `slurm.conf` | Shared: accounting, scheduling, logs | Same file |
| `site.conf` (included by `slurm.conf`) | `docker/`: five containers, fake GPUs, no cgroups | `node/`: one host runs every daemon, cgroups on |
| `gres.conf` | `docker/`: fake GPU files | `node/`: `AutoDetect=nvml` |
| `cgroup.conf` | `docker/`: disabled | `node/`: cores, memory and GPUs held to the job |
| `slurmdbd.conf` | `docker/`: test password | `node/`: password set on the node, never committed |
| `accounts.cfg` | `docker/`: team05 and team06, 4 GPUs each | `node/`: team01 to team12, no shares yet |
| `fake-nvidia-smi`, `fake-dmesg` | `docker/`: installed over the real tools | Not used |

`accounts.cfg` is loaded with `sacctmgr -i load file=accounts.cfg`. A second load changes nothing.

## Run

```
docker compose up -d --build
./test.sh
docker compose down -v
```

The first build downloads CPU PyTorch (about 200 MB). `test.sh` takes about 90 s.

## What `test.sh` checks

Accounts `team05` (user alice) and `team06` (user bob) each get a share of 4 GPUs (`GrpTRES=gres/gpu=4`). User carol has no account.

1. `accounts.cfg` loads both accounts with their 4-GPU share.
2. alice runs a job on `team05`.
3. alice is refused on `team06`.
4. carol is refused.
5. A two-node job runs on `c1` and `c2`.
6. alice's first 4-GPU job runs. Her second waits with reason `AssocGrpGRES`, and bob's 4-GPU job runs at the same time.
7. A `--requeue` job whose node goes down is requeued and starts again by itself, with `SLURM_RESTART_COUNT=1`.
8. `gpu_health.sh` passes on both nodes.
9. An Xid 79 planted on `c2` fails `c2` and not `c1`.
10. `launch.sh` runs as a 2-node batch job: health check, all-reduce, reduce-scatter, all-gather and all-to-all (2 CPU processes per node over gloo), Xid scan.
11. With a 92 °C GPU planted on `c2`, `launch.sh` stops before the collectives and names `c2`.
12. Chained restart. `checkpoint/restart.sh` trains `train_demo.py` on 2 nodes (2 CPU processes each, FSDP2) and saves every step. After step 4 is COMPLETE, `c1` goes down; Slurm requeues the job, and it starts again from the newest COMPLETE checkpoint. Its losses and its final weights must be bit-equal to a run that never stopped. The test brings `c1` back at once, as a spare node would, and skips Slurm's 120 s requeue wait.

## Faults by hand

The fakes read files in `/run/fake` on each node. For example:

```
docker compose exec c2 sh -c 'mkdir -p /run/fake && echo " Link 0: <inactive>" > /run/fake/nvlink'
docker compose exec slurmctld srun -N2 /lab/health/gpu_health.sh
docker compose exec c2 rm -rf /run/fake
```

| File | Replaces |
|---|---|
| `gpus` | `nvidia-smi --query-gpu` output, one line per GPU: index, temperature, uncorrected ECC errors, row remap pending, row remap failed |
| `nvlink` | `nvidia-smi nvlink -s` output |
| `dmesg` | the kernel log that `xid_scan.py` reads |

## Moving to the real node

1. Check the node's Ubuntu version. These configs are tested with Slurm 25.11. Slurm 23.11 (Ubuntu 24.04) may need changes.
2. Install `slurm-wlm slurmdbd munge mariadb-server`. Make the munge key on the node.
3. Copy `slurm.conf` and the `node/` files to `/etc/slurm/`. In `site.conf`, put the host name in place of `NODE`, and the output of `slurmd -C` in place of the `NodeName` line. Set the database password in `slurmdbd.conf`.
4. Load `node/accounts.cfg`, after adding each team's GPU share (TSAI sets it) and the users.
5. Run the health check and the NCCL test: `teams/07-infrastructure/health/TESTING.md`, level 3.

Docker cannot test these, so check them first on the node: GPU binding through cgroups, NVML autodetect, the munge key, and InfiniBand between nodes.
