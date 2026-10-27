# Slurm in Docker: two-account test

Week 1 test for the job queue (Team 7). It needs no GPU: each compute node has 8 fake GPUs that Slurm counts but does not bind to a device.

## Cluster

| Container | Role |
|---|---|
| `mariadb` | Accounting database |
| `slurmdbd` | Accounting daemon: accounts, users, limits |
| `slurmctld` | Controller |
| `c1`, `c2` | Compute nodes, 4 CPUs and 8 fake GPUs each |

Slurm 25.11 from Ubuntu 26.04. The Slurm 23.11 package in Ubuntu 24.04 cannot launch job steps in Docker, because it has no `CgroupPlugin=disabled`.

## Run

```
docker compose up -d --build
./test.sh
docker compose down -v
```

## What `test.sh` checks

Accounts `team05` (user alice) and `team06` (user bob) each get a share of 4 GPUs (`GrpTRES=gres/gpu=4`). User carol has no account.

1. alice runs a job on `team05`.
2. alice is refused on `team06`.
3. carol is refused.
4. A two-node job runs on `c1` and `c2`.
5. alice's first 4-GPU job runs. Her second waits with reason `AssocGrpGRES`, and bob's 4-GPU job runs at the same time.

## Different on the real node

- Real GPUs: `gres.conf` uses `AutoDetect=nvml`, and `cgroup.conf` turns cgroups on so that jobs are held to their GPUs.
- The munge key is made on the node and copied to each host. Here it is baked into the image.
- The database password is not `slurm`.
- TSAI sets the GPU share for each team per day, and we put it in `GrpTRES`.
