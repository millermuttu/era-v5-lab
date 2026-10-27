#!/bin/bash
# Two-account test for the Slurm queue. Run after `docker compose up -d --build`.
# team05 (alice) and team06 (bob) each get a share of 4 of the 16 fake GPUs.
# carol has no account. Exits non-zero on the first failed check.
set -u
ctl() { docker compose exec -T "$@"; }
as() { local u=$1; shift; ctl -u "$u" slurmctld "$@"; }
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; ctl slurmctld scancel -u alice -u bob >/dev/null 2>&1; exit 1; }

for i in $(seq 60); do ctl slurmctld sinfo -h -t idle -o %D 2>/dev/null | grep -qx 2 && break; sleep 1; done
[ "$i" -lt 60 ] || fail "c1 and c2 are not idle after 60 s"

# Accounts and users. "Nothing new added" on a rerun is fine.
ctl slurmctld sacctmgr -i add account team05,team06 GrpTRES=gres/gpu=4 >/dev/null || true
ctl slurmctld sacctmgr -i add user alice account=team05 >/dev/null || true
ctl slurmctld sacctmgr -i add user bob account=team06 >/dev/null || true

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
echo "All checks passed."
