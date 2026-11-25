#!/usr/bin/env python3
"""campaign/serve-wait-sample.py — where the engine's serve thread BLOCKS: samples /proc/<tid>/syscall at a fixed rate.

    .venv/bin/python campaign/serve-wait-sample.py <engine pid> <seconds> [hz=400]

A blocked thread yields no perf sample, so a perf profile shows what the serve thread RUNS and is silent on what it
WAITS for.  This is the complement: at a fixed rate it reads the thread's scheduler state and, when the thread is
blocked in a system call, the call's number and the user program counter at the call site, resolved with addr2line
on the engine binary (the pid's executable).  Output: state | syscall | call site | samples | share, sorted.
A diagnostic: never a tracker point.  The thread is the engine's main thread (tid == pid), which is the serve thread.
"""
import os, subprocess, sys, time
from collections import Counter

FALLBACK = {0: "read", 1: "write", 2: "open", 3: "close", 4: "stat", 5: "fstat", 7: "poll", 8: "lseek", 9: "mmap",
            11: "munmap", 16: "ioctl", 17: "pread64", 18: "pwrite64", 25: "mremap", 35: "nanosleep", 44: "sendto",
            45: "recvfrom", 61: "wait4", 72: "fcntl", 74: "fsync", 75: "fdatasync", 202: "futex", 217: "getdent64",
            230: "clock_nanosleep", 232: "epoll_wait", 235: "tgkill", 257: "openat", 270: "pselect6", 281: "epoll_pwait",
            288: "accept4", 292: "eventfd2", 318: "getrandom", 434: "pidfd_open", 435: "pidfd_send_signal", 436: "io_uring_setup"}


def main():
    pid, seconds = int(sys.argv[1]), float(sys.argv[2])
    hz = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
    exe = os.path.realpath(f"/proc/{pid}/exe")
    names = dict(FALLBACK)
    try:
        for line in open("/usr/include/asm/unistd_64.h"):
            p = line.split()
            if len(p) == 3 and p[0] == "#define" and p[1].startswith("__NR_"):
                names[int(p[2])] = p[1][5:]
    except OSError:
        pass
    c, pcs, n, t_end, dt = Counter(), {}, 0, time.monotonic() + seconds, 1.0 / hz
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        try:
            state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
            sc = open(f"/proc/{pid}/syscall").read().split()
        except OSError:
            break
        if sc and sc[0] not in ("running", "-1"):
            nr, pc = int(sc[0]), int(sc[-1], 16)
            key = (state, names.get(nr, f"sys_{nr}"))
            pcs.setdefault(pc, 0)
            c[(key, pc)] += 1
        else:
            c[("R" if sc and sc[0] == "running" else state, "(running)"), 0] += 1
        n += 1
        rest = dt - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)
    if not n:
        sys.exit("no samples (the thread is gone?)")
    sites = {}
    if pcs:
        out = subprocess.run(["addr2line", "-f", "-C", "-e", exe] + [hex(p) for p in pcs], capture_output=True, text=True).stdout.splitlines()
        for i, p in enumerate(pcs):
            sites[p] = out[2 * i][:96] if len(out) > 2 * i + 1 else hex(p)
    print(f"samples {n} over {seconds:.1f} s ({n / seconds:.0f}/s) of tid {pid} ({exe})")
    print(f"{'state':>5s} {'syscall':>16s} {'samples':>8s} {'share':>6s}  call site (user pc, addr2line)")
    for (key, pc), k in sorted(c.items(), key=lambda kv: -kv[1]):
        if k / n < 0.002:
            continue
        print(f"{key[0]:>5s} {key[1]:>16s} {k:8d} {100 * k / n:6.1f}%  {sites.get(pc, '')}")


if __name__ == "__main__":
    main()
