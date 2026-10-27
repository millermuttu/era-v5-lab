#!/bin/bash
# Usage: entrypoint.sh slurmdbd|slurmctld|slurmd
set -e
su -s /bin/sh munge -c munged
chmod 1777 /shared  # job output from any user
case "$1" in
  slurmdbd)  exec slurmdbd -D ;;
  slurmctld) until sacctmgr -n show cluster >/dev/null 2>&1; do sleep 1; done
             exec su -s /bin/sh slurm -c "slurmctld -D -i" ;;
  slurmd)    touch /dev/fakegpu{0..7}  # files behind the fake GPUs in gres.conf
             until scontrol ping | grep -q UP; do sleep 1; done
             exec slurmd -D -N "$(hostname)" ;;
  *)         exec "$@" ;;
esac
