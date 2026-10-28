# One image for every Slurm role. The munge key is made at build time,
# so all containers built from this image trust each other.
FROM ubuntu:26.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      slurm-wlm slurmdbd munge mariadb-client python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
# CPU torch for health/allreduce.py and checkpoint/train_demo.py, which fall back to gloo with no GPU.
# numpy: torch.distributed needs it to gather objects, which a multi-rank checkpoint save does.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir torch==2.9.1 numpy==2.5.2 --index-url https://download.pytorch.org/whl/cpu
ENV PATH=/opt/venv/bin:$PATH
RUN mkdir -p /etc/munge && dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key \
    && mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
    && chown munge:munge /run/munge && chown slurm:slurm /var/spool/slurmctld /var/log/slurm
# Test users. UIDs must match on every node, so they are baked in.
RUN useradd -m -u 2001 alice && useradd -m -u 2002 bob && useradd -m -u 2003 carol
COPY slurm.conf docker/site.conf docker/gres.conf docker/cgroup.conf docker/accounts.cfg /etc/slurm/
COPY --chown=slurm:slurm --chmod=600 docker/slurmdbd.conf /etc/slurm/
# Fakes for gpu_health.sh and xid_scan.py; faults are planted in /run/fake (see test.sh).
COPY --chmod=755 docker/fake-nvidia-smi /usr/local/bin/nvidia-smi
COPY --chmod=755 docker/fake-dmesg /usr/local/bin/dmesg
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
