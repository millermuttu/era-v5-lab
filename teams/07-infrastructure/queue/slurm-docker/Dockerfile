# One image for every Slurm role. The munge key is made at build time,
# so all containers built from this image trust each other.
FROM ubuntu:26.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      slurm-wlm slurmdbd munge mariadb-client \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/munge && dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key \
    && mkdir -p /run/munge /var/spool/slurmctld /var/spool/slurmd /var/log/slurm \
    && chown munge:munge /run/munge && chown slurm:slurm /var/spool/slurmctld /var/log/slurm
# Test users. UIDs must match on every node, so they are baked in.
RUN useradd -m -u 2001 alice && useradd -m -u 2002 bob && useradd -m -u 2003 carol
COPY slurm.conf gres.conf cgroup.conf /etc/slurm/
COPY --chown=slurm:slurm --chmod=600 slurmdbd.conf /etc/slurm/
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
