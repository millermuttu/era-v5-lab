# Testing the health check and the NCCL test

Run all commands from this folder. Levels 1 and 2 need only one GPU. Level 3 needs the shared node.

## 1. On one GPU

| Command | Expected |
|---|---|
| `python3 xid_scan.py --selftest` | `selftest ok` |
| `python3 xid_scan.py; echo $?` | `0` (no hardware Xid in your kernel log) |
| `EXPECTED_GPUS=1 ./gpu_health.sh; echo $?` | `<host> OK   1 GPUs`, then `0` |
| `./gpu_health.sh; echo $?` | `found 1 GPUs, expected 8`, then `1` |
| `SIZE_MB=64 torchrun --standalone --nproc-per-node=1 allreduce.py` | `sum correct -> PASS` |

If `xid_scan.py` exits `2`, it cannot read the kernel log. Add your user to the `adm` group, or run it as root.

## 2. Faked faults

These tests check that each fault is caught. They do not touch the hardware.

A hardware Xid (79: GPU fell off the bus):

```bash
printf 'NVRM: Xid (PCI:0000:3b:00): 79, pid=1\n' | python3 xid_scan.py -; echo $?
```

Expected: `HARDWARE GPU fell off the bus`, `drain this node`, then `1`.

A hot GPU with ECC errors, a failed row remap and a down NVLink link, from a fake `nvidia-smi`:

```bash
fake=$(mktemp -d)
cat > "$fake/nvidia-smi" <<'EOF'
#!/bin/bash
[ "$1" = nvlink ] && { echo " Link 0: <inactive>"; exit; }
echo "0, 92, 3, No, Yes"
EOF
chmod +x "$fake/nvidia-smi"
PATH=$fake:$PATH EXPECTED_GPUS=1 ./gpu_health.sh; echo $?
rm -r "$fake"
```

Expected: four `BAD` lines (temperature, ECC, row remap, NVLink), then `1`.
The fake prints the fields in the order `gpu_health.sh` asks for them: index, temperature, uncorrected ECC count, row remap pending, row remap failed.

A slow network, by setting a bandwidth floor that one GPU cannot reach:

```bash
MIN_BUSBW_GBPS=1 SIZE_MB=64 torchrun --standalone --nproc-per-node=1 allreduce.py
```

Expected: `-> FAIL` and exit code `1`. One GPU has no bus traffic, so its bus bandwidth is 0.

## 2b. Through Slurm, with no GPU

The Docker Slurm cluster in `../queue/slurm-docker` runs `gpu_health.sh` and `launch.sh` as real Slurm jobs on 2 fake nodes, with faults planted. See its README. One command: `docker compose up -d --build && ./test.sh`.

## 3. On the shared node (from 1 Nov)

```bash
srun -N1 --gres=gpu:8 ./gpu_health.sh    # each node by itself first
sbatch -N1 launch.sh                     # one node: NVLink only
sbatch -N2 launch.sh                     # two nodes: adds the network between them
cat nccl-allreduce-<jobid>.out
```

The job output ends with `PASS` or `FAIL`. On a failure, the lines above it name the host and the problem.

**Set the bandwidth floor.** Write down the `busbw` of a healthy run, for H100 and B200 separately and for one and two nodes. Set `MIN_BUSBW_GBPS` a little below it, for example 90 percent. Until then a slow node passes.

**A hang or low bandwidth.** Run again with `NCCL_DEBUG=INFO sbatch launch.sh`. The log says which transport NCCL picked: NVLink, InfiniBand or plain sockets. Plain sockets between nodes means InfiniBand is not set up.

**A failed node.** `launch.sh` does not drain nodes. Drain it by hand with `scontrol update nodename=<host> state=drain reason="<the BAD line>"`, then report it to TSAI.
