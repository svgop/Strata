# HOSTMEM LEG 0 — the law-17 BEFORE bank (aged box, 6 days uptime)

2026-11-07, 16:56–17:05 SAST, 3 boots per pack, standing launchers, zero env changes.
Battery: `campaign/batteries/hostmem-leg0.sh` (706a5716). Raw: `hostmem-leg0-{iq3s,iq2xs}.log`.

## The bank (3-boot means)

| pack | solo | pair | conc | vs standing bank |
|---|---|---|---|---|
| GSQ IQ3_S | 206.3 | 186.7 | 258.9 | −10% / −17% / −15% vs 229.1/224.8/303.5 |
| UD-IQ2_XS | 184.2 | 201.1 | 277.7 | −19% / −16% / −16% vs 226.5/240.0/329.7 |

The aged-box deficit, lawfully banked. These are the numbers every M1–M4 leg interleaves against.

## Page-class witnesses (per boot)

- **Gauge, engine down, every round (6/6): MADV_COLLAPSE rc=0, 100% promoted** (256 MiB → 262144 kB
  AnonHugePages). The 2 MiB supply exists the moment the engine's mass releases — restart-as-re-arm is
  now a measured fact, not a design claim. (Same probe with the engine up: ENOMEM, 0 folios.)
- **UD boots: the guard's free-2-MiB-block supply has CRASHED through the week**: 18 → 16 → 9 GiB (earlier
  boots) → **1 / 2 / 5 GiB** (these three). `compact_fail` climbed +7–14K per UD boot; +0/+0/+2 on GSQ boots.
- GSQ boots show no arena line at all — its organ is the FileExpertSource pinned complement (4K by
  construction), populated "through the file cache (buffered)" then page-locked.
- IQ2_XS identity verified: "experts loaded: 39.38 GiB" = the 39 GiB arena; shadow tier /dev/shm (before-state
  correct — no huge mount installed yet).

## Reading

The deficit (−10..−19%) sits on three 4K organs exactly as designed (HOSTMEM-NO-REBOOT.md). The engine-down
gauge at 100% while the guard's boot-time supply is 1–5 GiB pins the mechanism precisely: free memory exists
at boot, but the buddy's free lists are scattered (free ≠ order-9-coalesced). M1 removes the 38 GiB 4K shmem
organ (the largest single scatterer), M2 makes kcompactd re-coalesce between cycles, M3/M5 then decide the
arena route on supply evidence. Leg 1 = M1+M2 armed, same-day interleave against this bank.
