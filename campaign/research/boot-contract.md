# The Boot Contract — deterministic, verified, anchored boot state

Status: DESIGN (2026-11-18); evidence in F-382/F-383. Owner: engine (generate.cpp boot path, expert_cache.cpp).

## The defect, measured

Seven engine restarts on one host boot, one config, one model, one data set: M1 128.9–219.2 t/s.
The state is set at boot and is stable within a boot (a second leg on the bad boot read 133.3).
It is not the hardware (PCIe 5.0 x16, no throttle/ECC/XID, CPU at full speed — the 624 MHz cpufreq
reading is a nohzp_full artifact; see F-382).

**The master variable is the VRAM expert-cache hit rate, and it maps monotonically to every lane:**

| boot | M1 | hit | cpu-exp ms/window | window ms |
|---|---|---|---|---|
| 21:32 | 219.2 | 96.2% | 3.07 | 14.67 |
| 21:20 | 215.4 | 96.0% | 2.91 | 15.28 |
| 21:23 | 200.4 | 93.7% | 4.21 | 16.24 |
| 21:40 | 185.4 | 94.3% | 4.43 | 17.20 |
| 21:44 | 144.5 | 92.5% | 7.71 | 22.75 |
| 20:57 | 143.3 | 92.6% | 7.59 | 21.97 |
| 21:34 | 128.9 | 91.9% | 19.98 | 51.76 |

A 4.3-point hit-rate spread becomes a 50% throughput spread because **every lost hit pays the CPU
miss path** — and the corrupt-store rollback made that path ~2× dearer (Q2_0 store 75 µs/entry vs
native IQ3_S 152 µs; T66 anatomy). The system is at a knife edge: the hit rate is the only thing
standing between 219 and 129.

**Why the boot state is not reproducible.** The resident set = the profile fill's top-N + the adaptive
tier's swaps. Both float:
- The cache size is derived from *transient free VRAM* at boot (9,830–10,008 slots across boots; even
  `--expert-cache 9885` came up 9,970–9,971 — the engine adds headroom). N changes the fill's set.
- The adaptive tier swaps up to `--adapt-swaps 96` experts every 4 rounds on *decayed routing counts*
  from sampled text; there is no anchor, no budget, and no idle re-seed. A boot can settle into a
  resident set 3 points worse than its own fill and stay there (75k VRAM exchanges in the first leg
  vs 500 on a good boot).
- **The fill is verified at slot 0 only.** `STRATA_VERIFY_COMPLEMENT` (all-slot checksum) exists as a
  test hook, off by default. A degraded boot is invisible: /health says "ok", prewarm says "done".

## The contract (five refinements, in dependency order)

**R1 — ONE VRAM BUDGET OWNER.** The expert cache, batch slots, KV resident, vision reserve, draft head
and elastic arenas are today six flags that each subtract from free VRAM at boot (five owners of one
budget). Replace with one planner in the boot path: a named policy (`expert-cache auto` = target MiB
from the profile's coverage curve; `--expert-cache N` = exact N, no headroom) that logs the full
allocation table once. The canonical seed size becomes a function of the policy, not of what happened
to be free. Exit criterion: two boots log identical allocation tables.

**R2 — VERIFIED FILL, PUBLISHED FINGERPRINT.** Promote the checksum hook to the contract: verify a
deterministic spread (64 slots: every ceil(N/64)-th) *and* hash the resident table, publish
`boot fingerprint <hash> <slots>` in the log and `/health`. A fill that differs from the profile's
canonical prefix fails the boot loudly. Cost: ~90 MB D2H + a 24,576-entry table hash (< 100 ms).
Exit criterion: every boot's fingerprint is reproducible for a fixed policy (R1).

**R3 — ANCHORED RESIDENCY.** The profile's top-N is the anchor. The adaptive tier becomes a bounded
excursion: (a) a churn budget per request (`--adapt-budget` = ≤2× slots of swaps, default on);
(b) `--adapt-anchor-idle S`: after S seconds with no request, or at the first request after idle,
re-seed the resident set to the anchor (the existing fill path, re-run against the live table).
A bad boot can then never persist: the next idle returns the engine to 96%. Exit criterion: a bad
state cannot be reproduced after an idle re-anchor.

**R4 — BOOT CERTIFICATION.** After the fill (and after prewarm), run a fixed deterministic decode —
the probe's 65-token prompt, greedy, fixed seed, 64 tokens — and measure the hit rate and decode rate
from the engine's own counters. Compare to the contract (`hit ≥ 95.5%`, `M1 ≥ 200 t/s` on this box).
`/health` gains `"boot": {"state": "certified|degraded", "hit": …, "tps": …, "fingerprint": …}`.
The launcher prints the certification and refuses to call a degraded boot "ready" without saying so.
Exit criterion: no boot serves traffic without a certification line.

**R5 — ONE TELEMETRY SURFACE (simplification).** Today the boot facts live in four places: the engine
log, `/health` (prewarm), `/metrics` (windows), and the probe's parsing of old log lines. One boot
report object owns them: allocation table (R1), fingerprint (R2), certification (R4), and the tier
lines the probe currently scrapes. The probe and the GUI read the object, not the log.

## The proof protocol (campaign law 1/2/11)

- 5 boots of the standing recipe vs 5 boots of the candidate, interleaved ABAB in one hour.
- Per boot: one discarded leg, then the probe's M1/pair/conc; means over measured legs.
- Gate: every lane ≥ the standing mean AND the boot spread (max−min M1) inside noise (±4%).
- The prize measured today: bad draws cost 50%; medium draws 15%; stabilizing at the anchor moves
  the standing mean from ~185 to ~215 without touching the model.
- The store recovery (q2_0-rebuilt, incomplete) is the second, orthogonal lever: halving the miss
  cost shrinks the whole spread; it is an operator decision and is NOT part of this contract.

## Order of work

1. R2 (observability, no behavior change) — 1 day, proves R1's determinism claim.
2. R1 (budget owner) — 2 days, the determinism fix.
3. R5 (report object) — folded into R1/R2.
4. R4 (certification) — 1 day, the operator-facing guarantee.
5. R3 (anchor + budget) — 3 days, the behavioral fix; gated on R2's fingerprint to prove it.
