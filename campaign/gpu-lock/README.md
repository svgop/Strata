# strata-gpu-lock v3 — the floor-range design

v2's measured failure (2026-11-07): after 30 s of GPU quiet it HARD-locked 1800 MHz
(min = max), and the card cannot boost past a hard lock until the service's 3 s poll sees
load.  The clockwatch proof caught 1777-1792 MHz at 94-98% utilization — every burst from
a gap paid seconds at 57% clock, the uniform ~30% lane inflation and the live first-token
sluggishness the operator reported.

v3: two states; the HARDWARE owns the boost decision.  `-lgc 1800,3135` is a RANGE lock:
idle sits at the floor (never the lazy 210 MHz), boosting to max is a microsecond DVFS
step in the silicon — the service's latency leaves the boost path entirely.  This is
NVIDIA's recommended pattern for latency-sensitive serving (a full pin is for determinism
benchmarks).  Average power is LOWER than v2's busy-max.  Thermal guard: shed the floor at
83 C, resume at 78 C.  v1/v2 config names (SM_CLOCK/SM_MID) still honored.

## Install (root)
    sudo cp campaign/gpu-lock/strata-gpu-lock-v3 /usr/libexec/strata-gpu-lock
    sudo systemctl restart strata-gpu-lock
    # config (optional): /etc/sysconfig/strata-gpu-lock — SM_FLOOR/SM_MAX/TEMP_SHED/PORTS

## Verify
    journalctl -t strata-gpu-lock -f          # "clock floor 1800-3135" while an engine listens
    # during a generation, clocks should read 2900-3135 immediately, never stuck at 1780:
    nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader -l 1
