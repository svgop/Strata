# HOSTMEM day 1 — the wiring complete, first verdicts in (2026-11-07 evening)

Legs 0–3 banked across 16:56–20:41. Raw: `hostmem-leg0-*.log`, `hostmem-leg1-*.log`,
`hostmem-leg2-gsq.log` (the leg-3 run; the first leg-2 run's raw was overwritten by reuse — its
finding, the THP-only path, is preserved in commits 511bece6/e00b9c16 and the leg-2 design note).

## State of the three organs (all live before any reboot)

| organ | backing | evidence |
|---|---|---|
| UD-IQ2_XS arena 39.4 GiB | reserved pool | `hugetlb 2 MB pages` 3/3 boots (leg 1) |
| GSQ complement 37.6 GiB | reserved pool | `37.59 GiB of 37.59 GiB in 2 MiB pages (HUGETLB POOL)` (legs 3 arm boots) |
| shadow store 38 GiB | /dev/shm until the boot-armed pool (≥40000 pages) lifts the gate | pool-gated copy |
| pool | 24369/45056 (the aged buddy's runtime ceiling; compact_memory ×2) | sysctl.d 91 arms 45056 whole at next boot |

## Lane verdicts (ABAB same-window; the box drifts ±8% across windows, so same-evening pairs are the lens)

- **UD-IQ2_XS arena (leg 1)**: solo 193.2 vs 184.2 (**+4.9%**, adopt-grade); pair/conc flat; still −15% vs the
  standing bank → multi-organ deficit, store + full pool pending.
- **GSQ complement (leg 3, 2+2)**: arm solo 202.6 vs ctl 189.0 (**+7.2%**); pair flat (190.5 vs 190.0);
  conc 248.7 vs 260.2 (−4.4%, **ambiguous**: arm legs 238.9/258.4 straddle, 2+2 is underpowered for conc).
  vs the leg-0 bank: solo recovered to −1.8%, pair +2%, conc −4%.
- **GSQ control tax (leg 1 window)**: the partial-pool hold cost the un-migrated GSQ −12..−17% in its window;
  with the complement migrated (leg 3), same-window ctl reads ~190 solo — the tax is gone once the organ moves in.

## Holes found and fixed on the way (each by measurement)

1. T35 C2 asked THP, never the pool (0.01/37.67 accepted) → `huge_alloc` pool-first.
2. One-shot `cudaHostAlloc` bypassed the hugepage path (author's THP stall note) → the paced flip; the
   pool route registers in 13 s (pre-faulted 2 MiB pages — different physics, proven at 39.4 GiB).
3. First-come-first-served starved the hot organ (~90 KV host shadows × 0.26 GiB = 23.4 GiB in the boot's
   first 10 s) → pool priority at both faucets (≥8 GiB organs first; `STRATA_POOL_SMALL_ARENAS=1` opts in).

Named cost: GSQ boot 45 s → 93 s (paced step registration). Instrumentation: `ARENA PAGES` banner +
`STRATA_ARENA_PAGES=strict` (exit 70 on a silently-4K ≥1 GiB arena) standing in the binary.

## What decides the standing numbers (the definitive battery, post-boot)

One battery, 3+3 per pack, after the boot arms 45056 + the store moves to hugetlbfs: GSQ and UD-IQ2_XS
vs their banks (229.1/224.8/303.5 and 226.5/240.0/329.7), conc settled with 3+3 (tonight's −4.4% is
underpowered), store tier line + `HugePages_Free` draw as witnesses. Then the aging ratchet (cycle 1 vs N)
earns "never again", and the chart gets the recovery. IQ4_XS config (94 GiB need vs the 88 GiB pool) takes
its pool policy at its ladder leg — arena pool-backed, store on /dev/shm, or raise the reservation to 49152.
