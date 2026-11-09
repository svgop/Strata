# Performance setup — what the engine does automatically, and what the device needs

*Status: 2026-11-09. Every item below is measured on the campaign box and carries its evidence path.
The rule for this document: anything that can live in the engine does; anything device-level is
either installed by setup or detected and explained by `tools/strata-doctor.py`. No side scripts.*

## Automatic — ships with the engine (nothing to configure)

| guard / optimization | where | witness |
|---|---|---|
| **Swap guard**: the engine runs under a systemd user scope with `memory.swap.max=0` — its hot memory can never be zram-compressed (a swapped page is a decompression stall mid-window; measured: 299 MiB of the store swap-resident while decode read from storage) | every launcher (`STRATA_SWAP_GUARD=0` opts out; plain-exec fallback on non-systemd hosts) | `/proc/<engine-pid>/cgroup` shows `strata-engine-*.scope`; doctor **C7** |
| **Store mlock**: the shadow store mapping is locked so it is never swap-eligible (`STRATA_SHADOW_MLOCK=0` opts out) | `src/core/shadow_experts.cpp` | boot line `mlock refused` only on failure; store page class line |
| **Page-class truth**: every ≥1 GiB arena prints `ARENA PAGES: 2 MiB` / `4K - POOL UNARMED` (no pool = the default state) / `4K - DEGRADED CLASS` (a pool exists but this organ fell to 4K); benches can refuse a 4K boot (`STRATA_ARENA_PAGES=strict`, exit 70) | `src/core/pinned.cu` | boot banner |
| **Boot-time geometry gate**: the served context must match the armed geometry or the server refuses (exit 7) | `serve/server.py` + handshake | boot INFO banner |
| **CPU pinning** (optional): `STRATA_ENGINE_CPUS=8-15,24-31` pins the engine to one CCD on hybrid-cache CPUs; measured neutral on a quiet box — use it on contended machines | launchers, env-gated | `taskset` in the process cmdline |

## Device-level — one-time install, doctor-verified

### 1. GPU clock governor (`strata-gpu-lock` v6)
Pins **SM 3135 + MEM 14001 while any engine listens**, releases on exit; watchdogs, thermal shed,
event-reason logging. This is the measured record regime (means 216.3/208.2/290.1 at load 3–4; peaks
240.6/311.5 above the Nov-6 record; clocks 2962/13801 held 3/3 boots). MLPerf submissions pin both
domains at max for the same reason.
```
sudo cp campaign/gpu-lock/strata-gpu-lock-v6 /usr/libexec/strata-gpu-lock
sudo cp campaign/gpu-lock/strata-gpu-lock-v4.service /usr/lib/systemd/system/strata-gpu-lock.service
sudo systemctl daemon-reload && sudo systemctl enable --now strata-gpu-lock
```
Doctor **C8** verifies it. Rollback: stop the unit (`ExecStopPost` releases clocks).

### 2. CPU isolation for contended hosts (optional, measured mechanism)
The serial lane's 12.27 ms "GPU-reach wait" collapses to 0.98–1.55 ms on a quiet box — the engine's
doorbell poller loses its core to normal tasks when the machine is busy (the literature's measured
preemption penalty is 15.3 ms/launch-pair; the kernel tick adds 4 ms quanta). `isolcpus` + `nohz_full`
+ `rcu_nocbs` + `irqaffinity` make "quiet" permanent: the engine's CCD stops receiving system tasks,
ticks and interrupts. On a 9950X3D (two CCDs: 0–7 V-Cache, 8–15 frequency, SMT siblings +16):
```
sudo grubby --update-kernel=DEFAULT --args="isolcpus=8-15,24-31 nohz_full=8-15,24-31 rcu_nocbs=8-15,24-31 irqaffinity=0-7,16-23"
```
Pair it with `STRATA_ENGINE_CPUS=8-15,24-31`. Revert: `grubby --remove-args="..."`.

### 3. Kernel hygiene (experiments must not be silently changed)
Measured incident: PackageKit auto-updated the kernel mid-experiment (7.2.12→7.2.15). The census
exonerated that range (334 commits, none in relevant subsystems), but the practice stands:
```
# /etc/dnf/dnf.conf
excludepkgs=kernel*,kernel-modules*,kernel-modules-extra*,akmod-nvidia*
installonly_limit=6
gsettings set org.gnome.software allow-updates false
```
Log `uname -r` in every benchmark header (the battery scripts do).

### 4. Large-memory reservations — the doctrine that must not be violated
A standing 2 MiB hugetlb pool at 50–60% of RAM taxes **every** workload on the box: measured −12–15%
on our engine, and **−20–22% on the stock engine that never used a hugepage** (capacity loss, not page
class; an independent MoE deployment measured the same 0.966× tax with a 70 GiB pin). If a model
genuinely needs a pool (DeepSeek's all-or-none 110.85 GB copy), arm it **per session**, sized to the
measured worst draw + 2 GiB — never as a standing device state. The doctor's **C1** encodes the
doctrine: released is correct for the Qwen rotation; arm only before a DS session
(`campaign/results/ds-pool-topup-block.sh`), release after. The era's standing config
(`campaign/91-strata-hugepages.conf`, deleted 2026-11-09) must not come back — if any boot entry
still carries the args, strip them:
`sudo grubby --update-kernel=ALL --remove-args="default_hugepagesz=2M hugepagesz=2M hugepages=54272 hugetlb_free_vmemmap=on"`.

## Diagnostics that stayed

- **`tools/strata-doctor.py`** — 8 checks (pool vs doctrine, page class, supply, THP class, memlock,
  swap guard, governor, RAM floor), one-time explainers, paste-block fixes, JSON truth mode.
- **PCIe `L0_TO_RECOVERY_COUNTER`** — sampled flat (0/80 s) under probe load after the churn era; a
  standing witness for link instability if it ever grows during serving:
  `nvidia-smi pci -i 0 -gErrCnt`.
- **Battery protocol** — law 17 ABAB, prewarm-gated probes, witness lines per boot; scripts in
  `campaign/batteries/`; every result log in `campaign/results/`.
