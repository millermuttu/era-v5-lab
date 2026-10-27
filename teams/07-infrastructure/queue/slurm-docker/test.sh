#!/bin/bash
# Tests for the Slurm queue. Run after `docker compose up -d --build`.
# Part 1: team05 (alice) and team06 (bob) each get a share of 4 of the 16 fake GPUs; carol has no account.
# Part 2: the health scripts in teams/07-infrastructure/health run through the queue on the fake GPUs.
# Exits non-zero on the first failed check.
set -u
ctl() { docker compose exec -T "$@"; }
as() { local u=$1; shift; ctl -u "$u" slurmctld "$@"; }
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; ctl slurmctld scancel -u alice -u bob >/dev/null 2>&1; clear_faults; exit 1; }
# Plant a fault file (see docker/fake-nvidia-smi and docker/fake-dmesg) from stdin: fault c2 gpus
fault() { ctl "$1" sh -c "mkdir -p /run/fake && cat > /run/fake/$2"; }
clear_faults() { for n in c1 c2; do ctl "$n" rm -rf /run/fake; done; }

for i in $(seq 60); do ctl slurmctld sinfo -h -t idle -o %D 2>/dev/null | grep -qx 2 && break; sleep 1; done
[ "$i" -lt 60 ] || fail "c1 and c2 are not idle after 60 s"

# Accounts and users, from the same kind of file the real node loads. A rerun changes nothing.
ctl slurmctld sacctmgr -i load file=/etc/slurm/accounts.cfg >/dev/null 2>&1
out=$(ctl slurmctld sacctmgr -n -P show assoc where account=team05,team06 format=account,user,grptres | sort | tr '\n' ' ')
[ "$out" = "team05|alice| team05||gres/gpu=4 team06|bob| team06||gres/gpu=4 " ] \
  && pass "accounts.cfg loads team05 and team06 with 4 GPUs each" || fail "accounts: $out"

out=$(as alice srun -A team05 --gres=gpu:1 bash -c 'echo $SLURM_JOB_ACCOUNT' 2>&1)
[ "$out" = team05 ] && pass "alice runs a job on her own account" || fail "alice on team05: $out"

as alice srun -A team06 true >/dev/null 2>&1 && fail "alice ran on team06" || pass "alice is refused on the other team's account"
as carol srun true >/dev/null 2>&1 && fail "carol ran with no account" || pass "a user with no account is refused"

out=$(as alice srun -A team05 -N2 hostname 2>&1 | sort | tr '\n' ' ')
[ "$out" = "c1 c2 " ] && pass "two-node job runs on c1 and c2" || fail "two-node job: $out"

# GPU share: alice's second 4-GPU job must wait, while bob's job runs at the same time.
a1=$(as alice sbatch -A team05 --gres=gpu:4 --parsable -o /dev/null --wrap 'sleep 60')
a2=$(as alice sbatch -A team05 --gres=gpu:4 --parsable -o /dev/null --wrap 'sleep 60')
b1=$(as bob sbatch -A team06 --gres=gpu:4 --parsable -o /dev/null --wrap 'sleep 60')
sleep 3
state() { ctl slurmctld squeue -h -j "$1" -o '%T %r'; }
[[ $(state "$a1") == RUNNING* ]] && pass "alice job 1 (4 GPUs) runs" || fail "alice job 1: $(state "$a1")"
[[ $(state "$a2") == "PENDING AssocGrpGRES" ]] && pass "alice job 2 waits: team05 is at its 4-GPU share" || fail "alice job 2: $(state "$a2")"
[[ $(state "$b1") == RUNNING* ]] && pass "bob's job runs while team05 is full" || fail "bob job: $(state "$b1")"
as alice scancel "$a1" "$a2"; as bob scancel "$b1"

# Health scripts. They run as root, the way the infrastructure team runs them on the node.
clear_faults
out=$(ctl slurmctld srun -N2 /lab/health/gpu_health.sh 2>&1 | sort | tr '\n' ' ')
[ "$out" = "c1 OK   8 GPUs c2 OK   8 GPUs " ] && pass "gpu_health.sh passes on both nodes" || fail "health: $out"

printf 'NVRM: Xid (PCI:0000:3b:00): 79, pid=1, GPU has fallen off the bus.\n' | fault c2 dmesg
out=$(ctl slurmctld srun -N2 /lab/health/gpu_health.sh 2>&1)
grep -q "c2 BAD  hardware Xid" <<<"$out" && ! grep -q "c1 BAD" <<<"$out" \
  && pass "an Xid 79 on c2 fails c2 only" || fail "Xid fault: $out"
clear_faults

# launch.sh as a real 2-node job: 2 CPU processes per node instead of 8 GPUs.
launch() {
  local id
  id=$(ctl slurmctld bash -c 'cd /lab/health && GPUS_PER_NODE=2 SIZE_MB=8 ITERS=5 \
         sbatch --parsable --wait -o /shared/%x-%j.out launch.sh' 2>/dev/null)
  rc=$?
  log=$(ctl slurmctld cat "/shared/nccl-allreduce-$id.out" 2>&1)
}
launch
[ "$rc" = 0 ] && grep -q "sum correct -> PASS" <<<"$log" \
  && pass "launch.sh runs the health check, all-reduce and Xid scan on 2 nodes" || fail "launch.sh (rc $rc): $log"

for i in 0 1 2 3 4 5 6 7; do echo "$i, $([ $i = 3 ] && echo 92 || echo 40), 0, No, No"; done | fault c2 gpus
launch
[ "$rc" != 0 ] && grep -q "c2 BAD  GPU 3 is at 92 C" <<<"$log" && ! grep -q "all-reduce" <<<"$log" \
  && pass "launch.sh stops before the all-reduce when c2 has a hot GPU" || fail "hot GPU (rc $rc): $log"
clear_faults

echo "All checks passed."
