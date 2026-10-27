#!/usr/bin/env python3
"""Scan the kernel log of this node for NVIDIA Xid errors.

Usage: xid_scan.py [--since "YYYY-MM-DD HH:MM:SS"] [LOGFILE | -]
With no LOGFILE it reads dmesg, or `journalctl -k` when dmesg is restricted.
Exit 0: no hardware Xid. 1: a hardware Xid; drain the node. 2: no log to read.
"""
import argparse
import re
import socket
import subprocess
import sys
from collections import Counter

# Xids that point at the GPU or the node, not at the job (NVIDIA Xid catalog).
# Others, such as 13, 31 and 43, are usually a bug in the job and are only listed.
HARDWARE = {
    48: "double-bit ECC error",
    61: "micro-controller breakpoint",
    62: "micro-controller halt",
    64: "row remap failure",
    74: "NVLink error",
    79: "GPU fell off the bus",
    92: "high single-bit ECC rate",
    94: "contained ECC error",
    95: "uncontained ECC error",
    119: "GSP timeout",
    120: "GSP error",
    140: "unrecovered ECC error",
}
XID = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+)")


def kernel_log(since):
    for cmd in (["dmesg"], ["journalctl", "-k", "-q", "--no-pager"]):
        if since:
            cmd += ["--since", since]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pass
    return None


def scan(text):
    """Count Xids per (PCI bus id, Xid number)."""
    return Counter((pci, int(xid)) for pci, xid in XID.findall(text))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", help="kernel log file, or - for stdin")
    p.add_argument("--since", help="only read the live log from this time on")
    args = p.parse_args()

    host = socket.gethostname().split(".")[0]
    if args.log:
        text = sys.stdin.read() if args.log == "-" else open(args.log, errors="replace").read()
    else:
        text = kernel_log(args.since)
    if text is None:
        print(f"{host} BAD  cannot read the kernel log; add the user to the adm group or run as root")
        return 2

    found = scan(text)
    for (pci, xid), n in sorted(found.items()):
        kind = f"HARDWARE {HARDWARE[xid]}" if xid in HARDWARE else "job"
        print(f"{host} Xid {xid:>3} on {pci} x{n}: {kind}")
    if any(xid in HARDWARE for _, xid in found):
        print(f"{host} BAD  hardware Xid; drain this node")
        return 1
    return 0


def selftest():
    log = (
        "[  12.3] NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, GPU has fallen off the bus.\n"
        "[  12.4] NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, GPU has fallen off the bus.\n"
        "[  15.0] NVRM: Xid (PCI:0000:5e:00): 13, pid='<unknown>', Graphics Exception\n"
        "[  16.0] nvidia-nvlink: something that is not an Xid\n"
    )
    assert scan(log) == Counter({("0000:3b:00", 79): 2, ("0000:5e:00", 13): 1})
    assert scan("") == Counter()
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        sys.exit(main())
