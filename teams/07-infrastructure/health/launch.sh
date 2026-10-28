#!/bin/bash
# NCCL collective tests across nodes, with a GPU health check before and an Xid scan after.
# Submit from this folder: sbatch launch.sh   (one node: sbatch -N1 launch.sh)
# Pick H100 or A100 nodes with -p/--constraint; on 4-GPU nodes add --gres=gpu:4.
# Knobs: GPUS_PER_NODE (default: GPUs Slurm gave the job), PORT (default 29500),
#   OPS (default: all_reduce reduce_scatter all_gather all_to_all), TRACE_DIR (default: this folder),
#   and those of allreduce.py and gpu_health.sh.
#SBATCH --job-name=nccl-allreduce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
set -u
here=${SLURM_SUBMIT_DIR:?run with sbatch from the health folder}
gpus=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
# The health check counts the GPUs Slurm gave the job, even when the test uses fewer.
export EXPECTED_GPUS=${EXPECTED_GPUS:-${SLURM_GPUS_ON_NODE:-$gpus}}
start=$(date '+%F %T')
# NCCL flight recorder: on a hang, each rank dumps its last collectives to TRACE_DIR,
# which names the rank that stopped (Session 15). Needs a folder every node can write.
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${TRACE_DIR:-$here}/nccl_trace_${SLURM_JOB_ID}_rank_

echo "== health check on $SLURM_JOB_NODELIST"
srun --ntasks-per-node=1 "$here/gpu_health.sh" || { echo "FAIL: a node is unhealthy, see above"; exit 1; }

head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
rc=0
for op in ${OPS:-all_reduce reduce_scatter all_gather all_to_all}; do
  echo "== $op on $SLURM_NNODES nodes x $gpus GPUs"
  OP=$op srun --ntasks-per-node=1 torchrun --nnodes="$SLURM_NNODES" --nproc-per-node="$gpus" \
    --rdzv-backend=c10d --rdzv-endpoint="$head:${PORT:-29500}" --rdzv-id="$SLURM_JOB_ID-$op" \
    "$here/allreduce.py" || rc=1
done

echo "== Xids since $start"
srun --ntasks-per-node=1 python3 "$here/xid_scan.py" --since "$start" || rc=1

[ "$rc" = 0 ] && echo "PASS" || echo "FAIL"
exit "$rc"
