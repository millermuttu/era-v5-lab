#!/bin/bash
# Pre-flight check of the GPUs on this node. On every node: srun --ntasks-per-node=1 gpu_health.sh
# Prints one line per problem and exits 1 if the node should not run a job, 0 if it is fit.
# Knobs: EXPECTED_GPUS (default 8), MAX_TEMP_C (default 85).
set -u
host=$(hostname -s)
want=${EXPECTED_GPUS:-8}
bad=0
problem() { echo "$host BAD  $*"; bad=1; }

# [N/A] in a field means the GPU does not report it (no ECC, or no row remapping before A100).
if ! out=$(nvidia-smi --query-gpu=index,temperature.gpu,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,remapped_rows.failure \
                      --format=csv,noheader,nounits 2>&1); then
  echo "$host BAD  nvidia-smi failed: $out"
  exit 1
fi

n=$(grep -c . <<<"$out")
[ "$n" -eq "$want" ] || problem "found $n GPUs, expected $want"
while IFS=', ' read -r i temp ecc pending failed; do
  [[ $temp =~ ^[0-9]+$ ]] && [ "$temp" -gt "${MAX_TEMP_C:-85}" ] && problem "GPU $i is at $temp C"
  [[ $ecc =~ ^[0-9]+$ ]] && [ "$ecc" -gt 0 ] && problem "GPU $i has $ecc uncorrected ECC errors"
  [[ $pending == Yes || $pending == 1 ]] && problem "GPU $i has a row remap pending; reset the GPU"
  [[ $failed == Yes || $failed == 1 ]] && problem "GPU $i failed a row remap; replace the GPU"
done <<<"$out"

# NVLink on HGX A100/H100: a down link halves all-reduce bandwidth without any error.
down=$(nvidia-smi nvlink -s 2>/dev/null | grep -c inactive)
[ "$down" -eq 0 ] || problem "$down NVLink links are inactive"

# Xids since boot. ponytail: a GPU reset clears the fault but not the log; reboot or drain by hand.
python3 "$(dirname "$0")/xid_scan.py" || bad=1

[ "$bad" = 0 ] && echo "$host OK   $n GPUs"
exit "$bad"
