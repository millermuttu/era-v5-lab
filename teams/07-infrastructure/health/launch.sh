#!/bin/bash
# NCCL all-reduce test across nodes, with a GPU health check before and an Xid scan after.
# Submit from this folder: sbatch launch.sh   (one node: sbatch -N1 launch.sh)
# Pick H100 or A100 nodes with -p/--constraint; on 4-GPU nodes add --gres=gpu:4.
# Knobs: GPUS_PER_NODE (default: GPUs Slurm gave the job), PORT (default 29500), and those of allreduce.py and gpu_health.sh.
#SBATCH --job-name=nccl-allreduce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
set -u
here=${SLURM_SUBMIT_DIR:?run with sbatch from the health folder}
gpus=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
export EXPECTED_GPUS=${EXPECTED_GPUS:-$gpus}
start=$(date '+%F %T')

echo "== health check on $SLURM_JOB_NODELIST"
srun --ntasks-per-node=1 "$here/gpu_health.sh" || { echo "FAIL: a node is unhealthy, see above"; exit 1; }

echo "== all-reduce on $SLURM_NNODES nodes x $gpus GPUs"
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
srun --ntasks-per-node=1 torchrun --nnodes="$SLURM_NNODES" --nproc-per-node="$gpus" \
  --rdzv-backend=c10d --rdzv-endpoint="$head:${PORT:-29500}" --rdzv-id="$SLURM_JOB_ID" \
  "$here/allreduce.py"
rc=$?

echo "== Xids since $start"
srun --ntasks-per-node=1 python3 "$here/xid_scan.py" --since "$start" || rc=1

[ "$rc" = 0 ] && echo "PASS" || echo "FAIL"
exit "$rc"
