# The Reliability Stabilization Plan — a boot is deterministic or refuses to serve

Status: PLAN (2026-11-18). Extends `campaign/research/boot-contract.md` (R1/R2 landed 2026-11-18:
f384/f386; R4 exists as the external `campaign/boot-certify.sh`; R3/R5 unimplemented — verified:
no `--adapt-anchor*` / `--adapt-budget` in the tree). Owner paths named per phase.

Purpose: the campaign's purpose is performance; the November record shows the cost of a boot
whose state floats (f382: M1 128.9–219.2 on one host boot, hit 91.9–96.2% the master variable).
This plan closes every class of silent state variance so that perf work resumes on a floor that
cannot rot. Doctrine anchors: law 14 (nothing ships half-done), law 18/16 of the architecture
doctrine (one owner per capability, laws become structure), AGENTS "THE RECORD IS THE ENGINE".

## Phase 0 — close tonight's open ends (state, zero code)

1. **q2_0-rebuilt store: finish or delete** (f382 named it incomplete; f385's launcher
   VALIDATED-marker check stands). An incomplete store must not be armable. Owner: operator
   decision, one session.
2. **The uncommitted config line** (`config/custom-iq3_s-8081.json:54` pointing `--expert-profile`
   at the learned file): revert to the static profile until Phase 1's anchor policy exists.
   T18's rejection stands until a certified promotion path (Phase 1c) says otherwise.
3. **Root-owned state**: `data/expert-profile-learned.bin` is root-owned (root-run certify loop).
   Battery/certify scripts run as `svg`; anything writing repo state refuses to run as root
   (one guard in battery-lib.sh). Owner: campaign/battery-lib.sh.
4. **F-271 hygiene as code**: battery-lib pins/resets the anchor profile for both sides of every
   A/B automatically (today it is prose law; law 18 says make it executable).

## Phase 1 — R3 anchored residency (the behavioral fix; 3 days)

The adaptive tier becomes a bounded excursion around an anchor, not a free re-decision.

- **1a. Churn budget** — `--adapt-budget` (≤2× slots swaps per request, default on). A boot can
  no longer drift 3 hit-points below its own fill in the first leg (f382: 75k exchanges on a bad
  boot vs 500 on a good one). Owner: `src/core/expert_cache.cpp` (the adapt() path).
- **1b. Idle re-anchor** — `--adapt-anchor-idle S` (default 300 s): after S seconds with no
  request, or at the first request after idle, re-run the existing fill path against the live
  table to restore the anchor's resident set. A bad state cannot persist past one idle period.
  Owner: `src/program/generate.cpp` (the serve loop's idle branch, same site as the profile-save
  timer).
- **1c. Anchor ownership and promotion** — the anchor is a pinned, fingerprint-verified profile
  file. The `#477` learned file may ONLY become the anchor through an explicit promotion
  command: real-traffic soak → certification (Phase 2) → quality gate (MMLU canary, the T18
  floor) → copy to the anchor path + fingerprint recorded. The engine never auto-promotes and
  never writes the anchor (save path writes the learned file only). This kills the F-271
  poisoning class by construction. The server-side silent swap (`serve/server.py
  learned_profile_args`) is removed: the boot loads the anchor, full stop.
  Owner: `serve/server.py` + a `strata promote-profile` tool beside the launcher.

Exit criteria: (a) a deliberately corrupted resident set cannot survive one idle re-anchor
(hit returns to the anchor's class); (b) the f382 seven-restart protocol re-run: M1 spread
(max−min) inside ±4%; (c) F-271's synthetic-traffic poisoning replay cannot alter any boot.

## Phase 2 — R4 engine-level certification (1 day)

Move `boot-certify.sh`'s check inside the serve lifecycle so it is not a shell loop running as
root beside the engine:

- After prewarm reaches `done`, run one fixed greedy decode (the campaign M1 prompt, 0-temperature,
  64–256 tokens), read hit rate and t/s from the engine's own counters, publish in `/health`:
  `"boot": {"state": "certified|degraded", "hit": …, "tps": …, "fingerprint": …}` (the fingerprint
  R2 already prints). A degraded boot either refuses traffic (`"cert_refuse": true` config) or is
  labeled degraded in every surface. One retry before declaring degraded. Owner: `serve/server.py`
  (Prewarm's sibling lifecycle) + the engine's counter surface.
- `campaign/boot-certify.sh` shrinks to a wrapper that reads `/health` (kept for the launcher's
  exit code). Exit criterion: **no boot serves traffic without a certification line in /health.**

## Phase 3 — R5 one telemetry surface (1–2 days, folded with Phase 2)

One boot report object owns the boot facts (allocation table, fingerprint, certification, the
tier lines the probe scrapes today). The probe, the GUI and the tracker read the object; nothing
parses engine log lines for state. Owner: one JSON publisher in `serve/`, one emitter in the
engine boot path. Exit criterion: `grep`-based log scraping deleted from `campaign/probe-lane.py`.

## Phase 4 — integrity guards (the corrupt-pack class; 1–2 days)

The 2026-11-18 NaN hunt took a day because a corrupt canonical pack was invisible. Structure:

- **Pack/store checksums at boot**: the pack manifest gains per-file sha256 (full hash at build;
  size+mtime+sampled-chunk fast path at boot). A mismatch refuses the boot naming the file.
  Owner: the pack-manifest loader (already the one GGUF/pack reader — no second implementation).
- **Store completeness enforced by the engine**, not the launcher's VALIDATED marker alone.
- **Estate moves**: a `strata verify-estate` check that walks every config's referenced paths and
  hashes after any move; the 16:10 re-home class becomes a 30-second verification, not a
  boot-and-see. Owner: a tools/ script reading configs through the existing loader.

Exit criterion: a corrupted pack file produces a refused boot within one boot attempt, with the
file named — never a NaN in flight.

## Phase 5 — soak and tripwires (1 day + nightly, unattended)

- **Restart soak**: N=20 stop/start cycles, each certified, spread asserted ≤4%; runs nightly
  (costs ~20 × 64-token decodes). Owner: `campaign/battery-lib.sh` consumer, results to
  `campaign/results/soak-*.txt`.
- **Tracker gains a boot-spread column** (per `campaign/tracker/README.md` protocol); the incline
  law applies — spread must never widen across standing points.
- **Crash watch stays armed** (the P1 idle-SIGSEGV class, c45eeb2a1): any new crash blocks
  promotion by law, no exceptions.

## Phase 6 — performance resumes on the stabilized floor

With spread ≤4%, three-leg same-hour batteries carry verdicts that used to need five-boot hunts.
The queue, in priced order:

1. **R3's own prize** (measured today): stabilizing at the anchor moves the standing mean from
   ~185 to ~215 without touching the model (boot-contract.md's own pricing).
2. **The store recovery** (q2_0-rebuilt completion): +11/+20/+24% on/off means measured tonight;
   orthogonal to the contract — halving miss cost shrinks whatever spread remains.
3. The standing lever queue (tracker README, DS T53+ items, the W1 live-watch MMLU gate owed).

## Proof protocol for every phase (campaign laws 1/2/11)

Five standing-recipe boots vs five candidate boots, interleaved ABAB in one hour; per boot one
discarded leg then the M1/pair/conc probe; means over measured legs only; gate: every lane ≥
standing mean AND boot spread inside ±4%. Evidence to `campaign/results/`, journal F-entry +
commit in the same hour (law 14).
