# SUMMARY — persona-vector round, 2026-11-23

The question: the deployed model reads soft ("toy-like, barely trying") even under the F-283 WAR-01
max template. Can a control vector move the register — serious, assertive, resourceful — without
capability damage? This round closed the question for r02 (no, with mechanism), built r03 from a
fresh 72-boot capture, and found the first deployed-context WINDOW of the campaign.

## The chain, shortest form

1. **r02 falsified with mechanism** (`00-r02-verdicts.md`): 9-slot band add direction.18–26.
   Stock: 0.08 MIXED (serious 11/1 but ass null), 0.35/1.0 DAMAGE-FIRST (facts 1.00→0.69→0.00,
   words →18→5). Deployed 0.08: null (ass 9/7 p=0.40). Negative banked.
2. **Fresh capture** (`02-r03-capture/`): 72 boots, 3 traits × 12 cells × ±, exit 0.
   Pre-registered filters: **NO CANDIDATE, all traits** — prominence best 1.15 vs gate 1.25
   (diffuse in depth); stream-pair cos ≈ 0 in the band (decomposed across hyper-connection
   streams). Mechanism measured, threshold relaxation refused.
3. **Three external audits banked** (`campaign/research/2026-11-23-external-*.md`): methods
   (CAA/DIM/ITI/dose-scheduling; AxBench's "prompting beats steering" warning), low-level
   (**Hyper-Connections = 2409.19606**; final read-out = row-wise SUM of streams ⇒ our add gives
   ~**4× the read-out dose** a dense-model paper assumes — the r02 dose axis was mispriced, not
   the model), alternatives (SAE/decode/prompt/LoRA verdicts + weak-judge-proof eval upgrades).
4. **Router audit** (`04-r03-slot9/router-audit.md`): r02's band rows are router-invisible
   (0.99×) — r02's damage was read-out overdose, NOT routing. The slot-9 row is router-visible
   (2.21×, max 3.25 at blk 12). RARE (2608.21236) mechanism ported: P12 projection kills
   visibility (0.84/1.17 ≈ random) at 72.9% norm cost.
5. **Slot-9 falsifier** (`04-r03-slot9/VERDICT.md`): direction.8, single row, read-out-matched
   doses. **WINDOW at 0.3** (ass 10/3 p=0.046, ser 16/0 p=2e-05, facts 1.00/1.00) — serious holds
   **14/0 at |ΔW|≤12 words** (length confound closed). Knee: 0.6 first factual casualty (0.94).
6. **Deployed decision test** (`05-r03-deployed/VERDICT.md`): your template, max effort, thinking
   on, dose 0.3. **WINDOW: ass 12/4 p=0.038, ser 13/1 p=0.0009, dir 12/3 p=0.018, facts
   1.00/1.00, words +5%, trait density +40%.** First deployed-context window of the campaign.
7. **P12 knee test** (`06-r03-p12/VERDICT.md`): projected row at 0.6 — serious 18/0, facts
   1.00/1.00 where plain went 0.94. Router channel = the knee's carrier; projection validated as
   the dose-headroom path (single-run caveat banked; two-dose interleave queued).

## State of record

- **ADOPTED (default qwen standing, `10-adoption/VERDICT.md`)**: projected persona row
  (`…r03s9rnsp-s9-rho1.gguf:0.45`, direction.8, add, 8-8) as cvec table 1 + the standing refusal-r
  ablation as table 2 (project, 1-47) — composed by a two-table engine extension
  (`--control-vector-scaled2/--cvec-mode2/--control-vector-layer-range2`; parity-passed, gate-verified,
  deployed-verified: armed `add:8-8+project:1-47`, refusal battery ON 0/10, smoke assertive).
  Rollback: `engine/strata.pre-persona-merge` + drop the `*2` config block.
- plain row @0.3 deployed (`05-r03-deployed/`) stays the banked fallback dose.
- **PHASE-SCOPED (2026-11-24 afternoon, `12-phase/VERDICT.md`)**: the persona add lands only on the answer
  phase (`--control-vector-phase 248069:248068 --cvec-phase-table 1`; per-table gate, parity-proven).
  MMLU interleaved ON 66 / OFF 66 (exact tie, p=1.000; reasoning untouched, per-row length ratio 0.95)
  vs unscoped 65/66 at the same dose and 63/67 at 0.45. Register stronger: ass 13/3, ser 14/3, dir 14/1,
  facts 1.00 (WINDOW). The MC cost was a where/when effect, not the direction (cos(assertive, effort) =
  +0.013). Dose history: 0.45 adopted (0bd36187e) -> 0.3 (3c524a5dd) -> 0.3 phase-scoped (this).
- **Vector sources**: `capture/r03-run1/means.npz` sha16 `55be45efa4d6cef2`; elicit rows
  hash-pinned. Binary vectors in the model vectors dir, never in git.

## Re-open conditions / open queue

- DONE this session: the two-dose plain-vs-projected interleave (`07-legA-interleave/` — projection
  confirmed, adopted); the P12 deployed test at 0.45 (`08-legB-p12-deployed/` — WINDOW, adopted);
  judge controls (implemented in every AB, `judge_control_ok`); the refusal battery
  (`09-refusal-pair/`); the standing adoption (`10-adoption/`).
- The full performance gauntlet on the two-table boot (probe-lane ABAB + MMLU-Pro gate): the
  cvec cost profile barely changes (one extra launch at layer 8), but the law demands lanes, not
  arguments. Not yet run.
- Traits beyond assertive: effort/resourceful rows buildable from the same capture; composition
  requires the multi-trait orthogonalization check (VIAS warning: naive multi-value contrast
  collapses to a compliance signal).
- Non-vector lanes stay priced in the alternatives audit (prompt-search scored by trait-vector
  projection is the cheapest next lever; screened LoRA gated behind it).
- Scoped commit: `src/kernels/cvec.*`, `src/program/generate.cpp`, `src/prefill/prefill.cpp`,
  `src/core/layer.cpp`/`verify.cpp` (none), `src/kernels/cvec_parity.cpp`, `serve/handshake.py`,
  `campaign/vector-lab.py`, `campaign/vector-capture.py`, `campaign/vector-lab.md`,
  `campaign/results/vec-2026-11-23/` — never `git add -A` (the tree carries concurrent work).

## Run notes (the round's own laws, paid for in compute)

- `vector-lab.py ladder --out` resolves BARE names under the evidence root; deep relative paths
  nest. Regression fixed en route: ladder's `--seeds/--max-tokens` None-guard (deployed refactor).
- Never clean any path under a live run — one deployed run died at its final write this way
  (mkdir guard added; the compute was re-paid).
- Build CLI: `build --out <capture dir> --out-dir <vectors dir>`; refuses overwrite and slot<2
  (slot 1 not steerable — fail loud worked twice).
- Teacher-forced ±-contrast d′ is high everywhere (17–30): generic separability, never a vector
  quality claim. Peakedness/coherence filters and causal ladders are what make it evidence.
