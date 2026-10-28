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

# Automatic restart: a --requeue job on a node that fails comes back by itself, and SLURM_RESTART_COUNT
# tells it that it is a restart, so it loads the last COMPLETE checkpoint. Slurm waits 120 s before it
# starts a requeued job; the test skips that wait with starttime=now.
ctl slurmctld sh -c 'rm -f /shared/requeue-*'  # job ids start again when the controller is recreated
r=$(as alice sbatch -A team05 --requeue --parsable -o /dev/null \
      --wrap 'echo "run ${SLURM_RESTART_COUNT:-0}" >> /shared/requeue-$SLURM_JOB_ID; sleep 60')
for i in $(seq 10); do node=$(ctl slurmctld squeue -h -j "$r" -t R -o %N); [ -n "$node" ] && break; sleep 1; done
ctl slurmctld scontrol update nodename="$node" state=down reason=requeue-test
sleep 1
ctl slurmctld scontrol update jobid="$r" starttime=now
for i in $(seq 30); do [ "$(ctl slurmctld squeue -h -j "$r" -o %T)" = RUNNING ] && break; sleep 1; done
sleep 1
out=$(ctl slurmctld cat "/shared/requeue-$r" 2>&1 | tr '\n' ' ')
as alice scancel "$r"
ctl slurmctld scontrol update nodename="$node" state=resume 2>/dev/null  # ReturnToService=2 may have done it
[ "$out" = "run 0 run 1 " ] && pass "a job on a failed node is requeued and restarts by itself" || fail "requeue: $out"

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
[ "$rc" = 0 ] && [ "$(grep -c "result correct -> PASS" <<<"$log")" = 4 ] \
  && pass "launch.sh runs the health check, 4 collectives and the Xid scan on 2 nodes" || fail "launch.sh (rc $rc): $log"

for i in 0 1 2 3 4 5 6 7; do echo "$i, $([ $i = 3 ] && echo 92 || echo 40), 0, No, No"; done | fault c2 gpus
launch
[ "$rc" != 0 ] && grep -q "c2 BAD  GPU 3 is at 92 C" <<<"$log" && ! grep -q "== all_reduce" <<<"$log" \
  && pass "launch.sh stops before the collectives when c2 has a hot GPU" || fail "hot GPU (rc $rc): $log"
clear_faults

# Chained restart: a 2-node training job (2 CPU processes per node, FSDP2) loses a node, Slurm requeues
# it, and it resumes from the newest COMPLETE checkpoint with the same losses and final weights as a run
# that never stopped.
train() { ctl slurmctld bash -c "cd /lab/checkpoint && ROOT=/shared/$1 GPUS_PER_NODE=2 STEPS=10 $2 \
            sbatch --parsable -o /shared/$1.out $3 restart.sh"; }
until_file() { for i in $(seq "$2"); do ctl slurmctld test -e "$1" && return 0; sleep 1; done; return 1; }
ctl slurmctld rm -rf /shared/ref /shared/rs /shared/ref.out /shared/rs.out
train ref "" --wait >/dev/null
until_file /shared/ref/final.txt 5 || fail "reference training run: $(ctl slurmctld cat /shared/ref.out)"
r=$(train rs SLOW=1 "")
until_file /shared/rs/step_00000004/COMPLETE 90 || fail "restart run made no checkpoint: $(ctl slurmctld cat /shared/rs.out)"
ctl slurmctld scontrol update nodename=c1 state=down reason=restart-test
t0=$(date +%s)
for i in $(seq 10); do [ "$(ctl slurmctld squeue -h -j "$r" -o %T)" = PENDING ] && break; sleep 1; done
# The job needs 2 nodes, so it waits for c1: on the real cluster, a spare node or the repaired one.
ctl slurmctld scontrol update nodename=c1 state=resume 2>/dev/null  # ReturnToService=2 may have done it
ctl slurmctld scontrol update jobid="$r" starttime=now
until_file /shared/rs/final.txt 120 || { ctl slurmctld scancel "$r"; fail "restart run did not finish: $(ctl slurmctld cat /shared/rs.out)"; }
dt=$(( $(date +%s) - t0 ))
log=$(ctl slurmctld cat /shared/rs.out)
# Steps after the last COMPLETE checkpoint run twice; the last loss of each step counts.
same_loss=$(ctl slurmctld bash -c 'diff <(sort -n /shared/ref/losses.txt) <(tac /shared/rs/losses.txt | sort -nus -k1,1)' && echo yes)
same_weights=$(ctl slurmctld bash -c 'cmp -s /shared/ref/final.txt /shared/rs/final.txt' && echo yes)
grep -q "start at step [4-9], restart 1" <<<"$log" && [ "$same_loss" = yes ] && [ "$same_weights" = yes ] \
  && pass "a training job on a failed node restarts by itself in ${dt} s, with bit-equal losses and weights" \
  || fail "chained restart (loss $same_loss, weights $same_weights): $log"

echo "All checks passed."
