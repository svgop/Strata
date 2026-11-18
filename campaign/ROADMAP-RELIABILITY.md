# THE RELIABILITY ROADMAP — one plan, every finding, the road back to performance

Status: MASTER PLAN (2026-11-19). This is the single authoritative document consolidating:
the Boot Contract design (`campaign/research/boot-contract.md`), the stabilization plan
(`campaign/research/reliability-stabilization.md`), and research passes 1-3
(`proactive-reliability-research.md`, `reliability-research-deep.md`, `reliability-research-deep2.md`
— 60+ sources, every design decision traced to a measured precedent).

The engine's purpose is performance. This plan exists because the November record priced the
alternative: a boot whose state floats reads 128.9-219.2 t/s on identical inputs (f382), and
every debugging hour spent re-finding that is an hour not spent on the queue. The plan's own
success criterion is its own retirement: when it closes, the campaign goes back to kernels.

## The Contract (the invariants, from here on)

1. **A boot is deterministic or refuses to serve.** Same fingerprint in → same allocation
   table, same fill, same kernel plans, same certified decode — or a loud refusal.
2. **No masked errors.** Anything the engine survives silently, it must report loudly
   (fail-slow law: masking IS the defect).
3. **Hit rate is the page-first signal**, measured statistically (SPRT/CUSUM), never eyeballed.
4. **State is an owned artifact** — named, fingerprinted, version-salted, promotable only
   through an explicit gate; never a side effect of traffic.
5. **Recovery is the common path** — every start is a certified recovery, exercised nightly.

## Where we stand (2026-11-19)

| Piece | State | Evidence |
|---|---|---|
| R1 budget owner (exact N, pinned geometry) | ✅ LANDED | f386: 9600 slots exact, identical fingerprints two boots |
| R2 verified fill + fingerprint | ✅ LANDED | f384: spread check, refused boot on mismatch |
| R4 certification | ⚠️ EXTERNAL SHELL | `campaign/boot-certify.sh` (works, runs as root beside engine) |
| R3 anchored residency | ❌ NOT STARTED | no `--adapt-anchor*`/`--adapt-budget` in tree — **the critical path** |
| R5 one telemetry surface | ❌ NOT STARTED | probe still scrapes logs |
| Integrity guards (pack checksums) | ⚠️ LAUNCHER-ONLY | f385 VALIDATED marker; engine does not verify |
| Memory/pool architecture | ✅ STABLE | 106 GiB pool, complement, memfd store, CPU pin (2f1a8b828) |
| Known-good 220-class reachable | ✅ PROVEN | f385: 210.6/225.8 legs; f386 on2: 224.6/201.6/279.3 |
| Open wounds | q2_0-rebuilt incomplete; uncommitted config line 54; root-run state writes | f382, working tree |

## The state taxonomy (R3's design deliverable — from vLLM sleep-mode)

Every boot-time state object gets an explicit keep / discard / recompute decision:

| State class | Our object | On anchored restart |
|---|---|---|
| Parameters | expert blobs in VRAM slots | RESTORE from anchor (never re-decide) |
| Derived tensors | dequant scratch, plan tables | recompute (cheap, pure functions) |
| KV / conversations | session slots, conv cache | keep per existing lifecycle (T66) |
| Captured graphs | window graphs, cohort graphs | RESTORE descriptors, replay not recapture |
| Tier state | resident table, heat EWMA | RESTORE table; RESET heat to anchor priors |
| Learned state | learned profile file | WRITE-ONLY unless promoted (the gate, below) |

## PHASE 0 — close the open ends (hours, state only)

1. `config/custom-iq3_s-8081.json:54` uncommitted learned-file line: revert to static profile
   until Phase 1's anchor exists. 2. q2_0-rebuilt store: finish or delete (f385's launcher
   validation stands either way). 3. Guard: nothing writes repo state as root (battery-lib
   refuses uid 0). 4. F-271 hygiene as code: battery-lib pins/resets the anchor profile for
   BOTH sides of every A/B automatically.
*Exit:* clean tree, no root-owned state, every battery self-controls its anchor.

## PHASE 1 — R3 anchored residency (~3 days; the critical path)

**1a. Churn budget** — `--adapt-budget` (≤2× slots swaps/request, default on). A boot cannot
drift 3 hit-points below its fill in leg one (f382: 75k exchanges bad vs 500 good). The
fail-slow law behind it: background maintenance never recruits request-critical threads
without a hard budget (MongoDB WT-16718 pattern).
Owner: `src/core/expert_cache.cpp` (adapt path).

**1b. Idle re-anchor** — `--adapt-anchor-idle S` (default 300 s): after idle, re-run the fill
path against the live table. A bad state cannot outlive one idle period. Precedent: MoEpic's
idle-time cache reconfiguration (arXiv:2509.08342).
Owner: `src/program/generate.cpp` serve loop idle branch.

**1c. Anchor ownership and promotion** — the anchor is a pinned, fingerprint-verified profile.
Promotion pipeline (the ONLY path from learned → anchor): real-traffic soak → certification
(Phase 2) → MMLU gate (T18 floor 93.3) → copy + fingerprint record. The server-side silent
swap (`learned_profile_args`) is DELETED. Engine never writes the anchor. Curation prefers
domain-stable experts over raw-frequency ones (routing-consistency finding,
arXiv:2505.16056). Owners: `serve/server.py` + `strata promote-profile` tool.

**1d. PSI refresh policy** — PSI over per-expert activation-frequency deciles (anchor window
vs trailing), published each idle period: **<0.10 stable, 0.10-0.20 watch, ≥0.20 forces
re-anchor + drift journal** (credit-scoring thresholds). This replaces timer-based refresh
with distribution-evidence. Owner: engine idle path + Phase 3's report.
*Exit:* (a) corrupted resident set cannot survive one idle; (b) f382's seven-restart protocol
re-run: M1 spread ≤ ±4%; (c) F-271 poisoning replay cannot alter any boot; (d) offline
SCH/SRP measurement for Qwen3.8 prices anchor half-life (zero boots, uses public method).

## PHASE 2 — R4 certification inside the engine (~1 day)

After prewarm `done`: one fixed greedy decode. **Specification (computed, pass 2 §3):**
- SPRT on the per-entry Bernoulli hit stream — H0: hit ≥ 95.5% vs H1: ≤ 94.5%, α=β=0.05;
  ~346 routed entries/token → 64 tokens = ±0.27 pt CI (2.6× margin over the f382 gap);
  truncation at 128 tokens. Good boots certify in <64.
- **Greedy-decode output hash**: byte-equality against the anchor's recorded decode — the
  strongest canary (determinism-as-verifiability).
- `/health` gains `"boot": {"state":"certified|degraded","hit","tps","fingerprint","decode_hash"}`.
  Readiness FAILS CLOSED (K8s law): degraded boots refuse traffic (`cert_refuse` config) or are
  labeled degraded in every surface. One retry, then loud.
- `boot-certify.sh` shrinks to a wrapper reading `/health`.
*Exit:* no boot serves traffic without a certification line; single-instance lanes get the
fleet's warm-replica guarantee by proxy.

## PHASE 3 — R5 one telemetry surface (~1 day, folds into 2)

One boot report object owns: allocation table (R1), fingerprint incl. **kernel-plan
selections**, certification, tier lines the probe scrapes, PSI, and the drift alarms —
**CUSUM on hit EWMA in-boot** (thresholds derived from the hit-rate error budget; BOCPD if
CUSUM proves too blunt). Probe/GUI/tracker read the object; log-scraping is deleted.
*Exit:* `grep`-based state parsing gone from `campaign/probe-lane.py`.

## PHASE 4 — integrity guards (~2 days)

- **Pack checksums at boot**: sha256 at build in the manifest; size+mtime+sampled-chunk fast
  path at boot; mismatch = refused boot naming the file (the corrupt-pack day becomes 30 s).
  **DETECT_ONLY first** (InnoDB's lesson: measure corruption incidence before paying recovery).
- **Store writes doublewrite-shaped**: sequential scratch + fsync before in-place population.
- **Estate verify walk**: every config's referenced paths hashed after any move.
Owner: the pack-manifest loader (one reader, no second implementation).
*Exit:* a corrupted pack produces a refused boot, never a NaN in flight.

## PHASE 5 — soak and chaos, unattended (1 day + nightly)

Nightly restart soak: 20 stop/start cycles, each certified, spread asserted ≤4%. Four chaos
drills, each required to end REFUSED or CERTIFIED-SELF-HEALED, never silently slow:
corrupt-a-pack, kill-9 mid-window, pool-drain, profile-poison. Tracker gains a boot-spread
column under the incline law (spread never widens). Crash watch stays armed (P1 class).
*Exit:* a week of green soaks with zero manual intervention.

## PHASE 6 — soft checkpoint (post-stability, ~3 days)

The sleep-mode-class upgrade to R3: at idle, write the full anchored state (resident table +
graph descriptors + tier accounting + fingerprint salt); a same-config restart RESTORES it —
restart variance collapses to the fingerprint check (vLLM: wake is 18-20× a fresh load).
Address stability for everything graphs touch; re-seed what must differ (admission stamps,
session epoch — Firecracker's restore-uniqueness law). Hard checkpoint (NVIDIA
cuda-checkpoint+CRIU, driver ≥550, no UVM/IPC) remains the priced option beyond this.
*Exit:* restore-path restarts fingerprint-identical to the pre-stop boot, measured over 20 cycles.

## PHASE 7 — performance resumes (the point of all of it)

In priced order: (1) R3's own prize — standing mean ~185 → ~215 (measured, f382's pricing);
(2) q2_0-rebuilt store completion — +11/+20/+24% on/off means (f386); (3) next-layer expert
prediction (Pre-gated/AdapMoE class) to convert misses into prefetches; (4) MoEpic-style
split caching for hit-rate-per-GiB; (5) the standing lever queue (tracker README, DS T53+).

## The standing determinism rules (annex — binding on all future engine work)

1. **cuBLASLt**: untimed `heuristic[0]` is the default (pure function of shapes/lib/device/
   workspace); timed autotune winners load only from the fingerprint-keyed cache.
2. **Mapping-class law**: every mmap declares `MADV_RANDOM` (point reads) or
   `MADV_SEQUENTIAL` (sweeps); non-critical I/O (prewarm sweep, state saves) runs at
   `IOPRIO_CLASS_IDLE`; every mapping the prewarm derivation cannot see lands in
   `prewarm.extra_files` in the same commit (F-360 law).
3. **Fingerprint coverage**: profile, pack manifest hash, geometry, env, driver, kernel-plan
   selections — anything not hashed is not deterministic.
4. **Graph discipline**: stable scratch addresses; no allocator bookkeeping inside capture.
5. **State writes**: tmp + rename + fsync, always; anchors additionally version-salted.
6. **The canary**: byte-equality of the fixed greedy decode is the strongest signal we own —
   any change that breaks it is an output change, whatever the mechanism claims (F-230 law,
   now with teeth).

## Proof protocol (every phase; campaign laws 1/2/11)

Five standing boots vs five candidate boots, ABAB interleaved same-hour; one discarded leg
per boot, then M1/pair/conc; means only; gate: every lane ≥ standing mean AND boot spread
within ±4%. Evidence to `campaign/results/`, F-entry + commit same hour. The prize, priced:
bad draws cost 50%, medium draws 15% — stabilization moves the standing mean ~185 → ~215
without touching the model.

## The record

- Design: `campaign/research/boot-contract.md` (R1-R5, the estate's own)
- Plan v1: `campaign/research/reliability-stabilization.md`
- Research: `proactive-reliability-research.md` (principles) → `reliability-research-deep.md`
  (mechanisms + certification math) → `reliability-research-deep2.md` (fail-slow, PSI,
  kernel I/O, cuBLASLt rule)
- Evidence: f381-f386 (the boot-lottery arc), F-271 (poisoning), T18 (anchor quality gate)
