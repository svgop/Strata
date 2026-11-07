---
type: digest
id: preboot-readiness-audit
date: 2026-11-07
scope: pre-reboot systems-readiness audit for the 88 GiB huge-page memory-architecture install
box: Fedora 45, kernel 7.2.12-300.fc45.x86_64, 184 GiB RAM (MemTotal 193105792 kB), uptime 6 days
verdict: GO — no blockers; one managed post-boot verification, four managed degraded-path notes
---

# Pre-boot readiness audit — 2026-11-07 (one-shot reboot, 19:24 install)

## 0. Evidence timeline (journal, this boot)

- Box booted 2026-11-01 15:45, still running. The config install postdates the current boot by 6 days.
- 19:24:53 `sudo cp campaign/91-strata-hugepages.conf /etc/sysctl.d/` — installed.
- 19:24:53 and 19:25:10 `sudo tee /proc/sys/vm/nr_hugepages` — RUNTIME raise attempts.
- Result: `vm.nr_hugepages = 24369` of the 45056 ask (53%). HugePages_Total=24369, Free=24369.
- 19:25:18 `strata-huge-store.service` enable --now: mounted, active (exited), mount cap 40 GiB.
- Store copy: still `/dev/shm/strata-shadow-q2_0` (38 GiB, 11 files) — the `< 40000` gate in
  `campaign/shm-shadow.sh:29` correctly deferred (24369 < 40000), pool left for the arena.
- `/mnt/strata-huge` empty (0 bytes used) — tier-0 is armed-but-unpopulated until the first
  post-reboot copy.

The runtime partial reservation is the M5 doc's predicted runtime behavior (silent partial on a
fragmented 6-day buddy, shmem `Shmem: 45581184 kB` pinning low orders). It is evidence the runtime
re-raise path is unreliable — which is exactly why this reboot is the move. It is NOT evidence the
boot path fails: at sysinit the buddy is pristine.

## 1. Ordering — no race (benign)

Chain (all verified from unit files, not assumed):

1. `systemd-sysctl.service`: `Before=sysinit.target`, `DefaultDependencies=no` → applies
   `/etc/sysctl.d/91-strata-hugepages.conf` (and 90) at sysinit, before any service with default
   dependencies. `ConditionPathIsReadWrite=/proc/sys/net/` holds. `TimeoutSec=90s` — a pristine-buddy
   45056-page allocation is seconds.
2. `strata-huge-store.service`: no `DefaultDependencies=no` → implicit `After=sysinit.target`;
   `After=multi-user.target` + `WantedBy=multi-user.target` → runs at end of boot. Strictly after (1).
   No fstab entry for `/mnt/strata-huge` (grep: none) → no fstab-generator double-mount race.
3. No engine autostart exists: no crontab `@reboot`, no `~/.config/autostart` entries, no user
   systemd units, no rc.local. Every engine launch is interactive, post-multi-user.target.

So: sysctl → mount → login → launcher. No unit in the transaction can see a partial pool while
holding a mount, and no launcher can run before both. Name-the-fix not needed.

## 2. Boot-time allocation risk — low (managed)

- Ask: 45056 × 2048 kB = 92274688 kB = 47.8% of MemTotal (193105792 kB).
- E820 device-reserved regions this boot sum well under 1 GiB (largest 0x7ab7f000–0x97cfefff
  ≈ 464 MiB); kernel image + initramfs release before sysinit; no `hugepagesz=`/`hugepages=` on the
  cmdline (read: `/proc/cmdline`) — the entire ask comes from the sysctl path on a near-empty buddy.
- Kernel behavior if short: `nr_hugepages` silently partials (proven live today at 24369). No boot
  failure is possible from a short pool — the sysctl write itself cannot fail the unit.

Degradation ladder if the boot pool comes up short (every rung verified in source, none gates
multi-user.target — the store unit is `Wants`, not `Requires`/`BindsTo`):

1. Pool < 40000 → `shm-shadow.sh:29-31` exits 0 (planned deferral) → launcher `|| true`
   (`launch-custom-iq3s-8081.sh:116`) → tier guard falls to `/dev/shm` (empty on first boot) →
   disk canonical `/home/svg/LLM/data/packs/q2_0` (present, 11 files, 40303644672 B ≈ 37.55 GiB).
   Lane boots; CPU-miss path page-cached (slower, byte-identical).
2. Engine arena: `src/core/pinned.cu:252` pool-priority gate (≥ 8 GiB draws pool; small arenas
   skipped unless `STRATA_POOL_SMALL_ARENAS`) → `MAP_HUGETLB` mmap is all-or-nothing (reservation
   at mmap time) → short pool fails the mmap cleanly → fallback 2 MiB-aligned anon + `MADV_HUGEPAGE`
   (THP is `madvise` on this box — verified `/sys/kernel/mm/transparent_hugepage/enabled`;
   no `always`-mode reclaim storm) → `ARENA PAGES: 4K - DEGRADED CLASS` banner announces it
   (`pinned.cu:431`). No strict flag armed in any launcher (`STRATA_ARENA_PAGES` unset → no exit 70).
3. Deadlock audit: store script has no waits or locks (mkdir -p, mountpoint -q, rsync, exit);
   launcher swallows its non-zero exits; engine fallback is a synchronous mmap failure. Nothing
   spins, nothing blocks boot.

Also verified: with a fully armed pool, post-boot MemTotal ≈ 193105792 − 92274688 ≈ 100831104 kB
(~96 GiB) → `/dev/shm` default tmpfs ≈ 48 GiB — even the degraded `/dev/shm` store tier still fits
(37.55 GiB + 8 GiB headroom check passes).

## 3. tmpfs wipe inventory — nothing else dies (benign)

- `/dev/shm`: `strata-shadow-q2_0` (38 GiB, 11 files — matches the source pack file count) + one
  4 KiB lttng-ust sentinel. Nothing else.
- `/run/user/1000`: pure desktop-session sockets/pipes (pipewire, wayland, gnome, flatpak) — all
  regenerated by the session. Largest real dir is 2 MiB (`doc`).
- Rebuild path: launcher-automatic (`launch-custom-iq3s-8081.sh:116` → `campaign/shm-shadow.sh`),
  source on DISK at `/home/svg/LLM/data/packs/q2_0` — verified present (experts.bin 33973862400 B,
  dense.bin, embd.bin, tokenizer/; 11 files). Survives reboot. The task's "data/packs/q2_0" is
  `/home/svg/LLM/data/packs/q2_0`, NOT inside the repo.

## 4. gpu-lock v4 — boots clean (benign)

- Enabled, active (since 16:24 today), `/usr/libexec/strata-gpu-lock` in place.
- `Type=notify` + `WatchdogSec=30`: script pings `systemd-notify WATCHDOG=1` every POLL=3 s
  (verified in the script) — dead-man holds.
- `ExecStartPre=-/usr/bin/nvidia-smi -i 0 -rgc` / `-rmc`: leading dash = failure-tolerant,
  idempotent at every boot.
- `ExecStopPost` releases on every stop path; `Restart=on-failure` heals hangs.
- `Wants=nvidia-persistenced.service` (enabled + active) — Wants is non-fatal if it ever regresses.
- Engine ordering: none needed. The state machine polls `PORTS="8080 8081"` with GRACE=45; engine
  start is detected within one poll. Clock lock is perf-only; no correctness dependency.

## 5. zram — no change required (benign)

- `/usr/lib/systemd/zram-generator.conf`: `zram-size = min(ram, 8192)` → 8 GiB, lzo-rle, priority 100,
  sole swap device. Post-reboot RAM (~96 GiB general) keeps min() at 8192 — size unchanged.
- `vm.swappiness=10` is owned by tuned (`/usr/lib/tuned/profiles/throughput-performance/tuned.conf`,
  `[sysctl]` section) — tuned is active, applies after systemd-sysctl, and carries NO hugepage or
  compaction keys. Swappiness 10 survives the reboot. Nothing to persist, nothing to change.
- Swap is nearly full today (7.9/8 GiB) — dies with the reboot; zram rebuilds empty. That pressure
  was the degraded 6-day state, not a config gap.

## 6. Conversation slots — disk-backed, survive (benign)

- `STRATA_CONV_SLOT_DIR` default `/home/svg/LLM/data/conversation-slots`
  (`launch-custom-iq3s-8081.sh:33`) — on the home filesystem, verified present with 5 drained slots
  (slot-10/11/12/13/30, ~240 MiB each, drained 20:41 tonight). No tmpfs dependency in the restore
  path (engine reads the dir at boot RAM-preflight).
- Caveat, not a boot risk: drain happens at clean engine exit. A hard-kill reboot loses unparked
  conversations → fallback is the pre-T66 re-read behavior (slower start, no failure).

## 7. First launcher after reboot — order traced (managed)

`launch-custom-iq3s-8081.sh` sequence: tier guard (lines 65-80) resolves BEFORE
`shm-shadow.sh` (line 116) runs. First start post-reboot: tier-0 empty (mount fresh), `/dev/shm`
wiped → guard picks the disk canonical pack → engine serves from page cache; the copy to
`/mnt/strata-huge` runs during the same launch (~37.55 GiB, NVMe → RAM-backed fs; the /dev/shm
rebuild measured 12 s, expect the same order of magnitude — it blocks that one launcher
invocation, NOT boot). Next restart picks tier-0. To have tier-0 on the FIRST start instead:
`bash campaign/shm-shadow.sh` before launching (gate passes at pool 45056 ≥ 40000; expected pool
draw ≈ 19226 pages, leaving ≈ 25830 free for the arena).

Pool cover math at full arm: store 37.55 GiB (≈19226 pp) + GSQ complement 37.6 GiB (19251 pp) ≈
38477 pp of 45056; UD-IQ2_XS variant 39.4 GiB (20176) → 40402 pp. Both fit with ≥ 4500 pp slack.
KV/slot arenas (~3 GiB) are pool-priority-skipped by default (`pinned.cu:252`). Mount cap check:
40 GiB cap ≥ 37.55 GiB + 2 GiB headroom (script line 38) — passes with ~0.5 GiB margin at check
time; real usage ≈ 37.6 GiB leaves ~2.4 GiB on the mount. Only the store lives on the mount.

## 8. Risk register

| # | Risk | Grade | Basis |
|---|------|-------|-------|
| 1 | Boot pool partial (< 45056) | managed | silent-partial proven at runtime; boot buddy pristine; ask is 47.8% of RAM; full degradation ladder verified in source, none fails boot |
| 2 | First start serves disk tier | managed | guard-before-copy order; one restart (or pre-run shm-shadow.sh) picks tier-0 |
| 3 | Mount cap margin (40 vs 37.55+2) | managed | passes by ~0.5 GiB at check; 2.4 GiB real headroom; store is the only tenant |
| 4 | Interrupted copy poisons tier-0 guard | benign | `-f experts.bin` presence guard would accept a partial store; window needs pool exhaustion mid-copy (impossible on first boot; standing practice: run shm-shadow with engine down); fast-path size verify catches it on the following start |
| 5 | Hard-kill reboot loses undrained slots | benign | disk restore path intact; loss = slower start, not failure |
| 6 | tuned interference with sysctls | benign | profile carries only swappiness + somaxconn; no hugepage/compaction keys |
| 7 | Ordering race | benign | sysinit → multi-user chain proven from unit files; no autostart launchers |
| 8 | gpu-lock boot failure | benign | enabled+active, watchdog-pinging, dash-guarded idempotent pre-starts, non-fatal Wants |

## 9. GO/NO-GO checklist (run in order; each with verification + one-line rollback)

| # | Check | Command | Expect | Rollback |
|---|-------|---------|--------|----------|
| 1 | PRE: config files owned | `sysctl --system 2>&1; sysctl vm.nr_hugepages vm.compaction_proactiveness` | no errors; 45056 / 40 on a quiet box (partial acceptable NOW — boot re-arms) | `sudo rm /etc/sysctl.d/91-strata-hugepages.conf /etc/sysctl.d/90-strata-vm.conf` |
| 2 | PRE: canonical pack on disk | `find /home/svg/LLM/data/packs/q2_0 -type f \| wc -l` | 11 | — (read-only) |
| 3 | PRE: slots on disk | `ls /home/svg/LLM/data/conversation-slots/` | slot-*.bin present | `STRATA_CONV_SLOT_DIR=` (launcher-level) |
| 4 | PRE: units enabled | `systemctl is-enabled strata-huge-store strata-gpu-lock` | enabled, enabled | `sudo systemctl disable --now strata-huge-store` |
| 5 | POST: pool armed | `grep HugePages_Total /proc/meminfo` | 45056 (≥ 40000 minimum to arm tier-0) | `sudo sh -c 'echo 45056 > /proc/sys/vm/nr_hugepages'` (steps, engine down) or accept degradation |
| 6 | POST: compaction tuned | `sysctl vm.compaction_proactiveness` | 40 | `sudo sysctl vm.compaction_proactiveness=20` |
| 7 | POST: mount up | `mountpoint /mnt/strata-huge && stat -f -c %T /mnt/strata-huge` | mounted, hugetlbfs | `sudo systemctl restart strata-huge-store` |
| 8 | POST: swappiness (tuned) | `sysctl vm.swappiness` | 10 | `sudo sysctl vm.swappiness=10` (runtime) |
| 9 | POST: gpu-lock alive | `systemctl is-active strata-gpu-lock && journalctl -u strata-gpu-lock -b --grep WATCHDOG -o cat \| tail -1` | active; watchdog pings | `sudo systemctl restart strata-gpu-lock; nvidia-smi -i 0 -rgc; nvidia-smi -i 0 -rmc` |
| 10 | POST: store tier armed | `bash campaign/shm-shadow.sh` | "tier /mnt/strata-huge … 2 MiB by construction (pool draw ≈19226 pages)" | `rm -rf /mnt/strata-huge/strata-shadow-q2_0` (falls back to /dev/shm/disk) |
| 11 | POST: first launch class | launcher stderr | `launcher: shadow store /mnt/strata-huge/strata-shadow-q2_0` + `ARENA PAGES: 2 MiB`, NO `4K - DEGRADED CLASS` | `STRATA_NO_LARGEPAGES=1` per-run A/B, or `STRATA_SHADOW_EXPERTS=""` shadow rollback |
| 12 | POST: MemTotal accounting | `grep MemTotal /proc/meminfo` | ≈ 100831104 kB (~96 GiB) | — (informational) |

## 10. Verdict

GO. Reboot. The only genuinely open item is empirical: whether the pristine boot buddy grants all
45056 pages (checklist #5). Every branch of "it didn't" is already built, verified in source, and
boot-safe — worst case is the documented degraded class with the banner announcing it, which is
precisely what this install exists to eliminate.
