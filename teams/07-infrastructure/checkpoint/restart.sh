#!/bin/bash
# Training job that Slurm restarts by itself: a node failure requeues it, and train_demo.py resumes
# from the newest COMPLETE checkpoint in ROOT. Submit from this folder: ROOT=/shared/run1 sbatch restart.sh
# Slurm starts a requeued job after about 120 s. Knobs: ROOT (required), GPUS_PER_NODE, PORT, and those of train_demo.py.
#SBATCH --job-name=restart
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
set -u
here=${SLURM_SUBMIT_DIR:?run with sbatch from the checkpoint folder}
: "${ROOT:?set ROOT to a folder every node can write}"
gpus=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "== $(date '+%F %T') start ${SLURM_RESTART_COUNT:-0} on $SLURM_JOB_NODELIST"
# A new rendezvous id on each restart, so a requeued job never meets the old one.
srun --ntasks-per-node=1 torchrun --nnodes="$SLURM_NNODES" --nproc-per-node="$gpus" \
  --rdzv-backend=c10d --rdzv-endpoint="$head:${PORT:-29500}" --rdzv-id="$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}" \
  "$here/train_demo.py"
