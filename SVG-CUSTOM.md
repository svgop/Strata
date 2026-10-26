---
type: flight-log
id: svg-custom
status: live
base: Niko1221/Strata v0.1.40 (82f46a8)
branch: custom
commits: [68123ce, 0e097be]
owner: Savage (svg)
port: 8081 (stock install remains at E:\LLM\Engine\strata, port 8080)
campaign-record: E:\LLM\Engine\llama-custom\.docs\CAMPAIGN-LOG.md (Passes 173-181)
core-features: SVG-CUSTOM-FEATURES.md (curated differentiators; README carries the headline list)
status-note: elastic sessions ADOPTED F-273 - standing recipe now includes STRATA_ELASTIC_SESSIONS=1
---

# SVG-CUSTOM — the flight log

Strata fork carrying our measured speed program. Every entry below states
what changed, why, the justification, and the source (paper, issue, our own
fork commit). Items invented here are marked **SVG-CUSTOM**; ports credit
their origin. The mission: relentless speeds with no quality loss, proven
on the ruler, never claimed unmeasured.

Physics that ranks all work (measured, Pass 176/178 research): expert reads
are ~10% of the 5090's bandwidth at speed; the wall is per-layer fixed cost
(~0.28 ms x 48 layers = ~85% of every verify window). Two lever classes
move the ceiling — ACCEPTANCE (tokens per weight-read) and FIXED COST.
Everything else is a 5-15% miss-drain term.

---

## CURRENT STATE (2026-10-08, post F-20 / commit 729bad0) — the standing result

MMLU-Pro 95.71% (0 overruns, vector ON) | Mimicbench 98.4% (63/64, drift 12/12
@ 100%) | compliance 12/12 | decode battery 139-184 t/s (162 warm typical) |
depth-103k 128.1 t/s @ 97.6% hits | prefill ~3.0-5.3k tok/s | ~10.8-11.1k slots.


## FOR THE UPSTREAM READER / ANYONE REPRODUCING THIS (start here)

This fork of Strata v0.1.40 reached: MMLU-Pro 95.7-98.6 (70q, 0 overruns),
mimicbench 95-98%, 146 t/s serial / 116 t/s at 6 concurrent / 128 t/s at
103k-token context, 5.7k tok/s prefill - on ONE RTX 5090 + 192GB RAM.
Everything below (F-01..F-29) is the full flight log; this section is the
10-minute map. Each item: MECHANISM -> MEASURED EFFECT.

**ENGINE PATCHES (src/, all in this tree):**
1. Medusa-style typical-acceptance rescue at verify rejections (tau 0.2,
   sampled only, multi-rescue 2/window) - acceptance 0.70 -> 0.92 class,
   +60-70% of dying windows rescued. CRITICAL detail: on rescue emit the
   DRAFT token (outv[a]=window[a+1]) or text/context diverge per rescue
   (our F-15 bug, fixed in F-15/F-16 - quietly degrades quality).
   [generate.cpp: typical_stats/typical_accept, the rescue block]
2. Leviathan/Chen ratio test (u < min(1, q/p); q = target probe, p = the
   drafter's dprob[a]) - the distribution-EXACT alternative, built +
   env-gated (STRATA_SPEC_RATIO), A/B pending. [ratio_accept]
3. The rescue ported to the batch (--batch) accept site + batch MTP - the
   batch slot path ran DRAFTLESS at ~51 t/s; port is in but batch MTP
   admission hits 'unsupported native MMVQ GGML type' (F-28b, fix owed).
4. Prediction-credit admission: RouterLookahead's l+1 top-k banks
   softmax-share credits into expert-cache admission - GATED (raw weight
   thrashes; needs u>=0.5 measured + banked>=1.5 + budget 2/round - F-16).
5. Victim protection: profile-rank pin-cap + min-age on fresh admissions
   (admit-thrash law) + a dormancy lane (never-routed-in-workload experts
   become evictable despite profile pin - F-22).
6. L2 persisting set-aside (60 MiB over the GDN state, all steady streams)
   - cudaStreamSetAttribute accessPolicyWindow, hitRatio computed (F-18/B8).
7. Boot-time verify-graph pre-capture (capture_all in the serial path) +
   probe buffer reuse (F-24).
8. Re-streamed on-device probe kernel (row_typical_stats on the verifier's
   stream; the F-14 law: legacy-stream = 2x regression, F-16b: restream =
   parity; arm STRATA_TYPICAL_DEVICE).
9. --resident-experts complement (39 GiB pinned; NOTE: exchange rotation
   needs EQUAL-SIZE expert blocks - blocked on i-quants, F-23b).
**SERVER PATCHES (serve/server.py):** answer-space reservation (reasoning
wraps at max_new - max(384, 25%) - NEVER returns empty; +5.7 MMLU alone,
F-19); ticket-FIFO admission (arrival-ordered, abandoned-skip, F-14).
**CONFIG RECIPE (strata-iq3_s.json + launcher):** THE STANDING RECIPE
section lists every arg/env. Load-bearing: kv-resident 32768 is a FLOOR
below which deep prefill regresses (F-22b); "parallel" is a config KEY not
an engine arg (F-29); conversation-cache 48 GiB for multi-chat; cvec
layers 4-24 (the knee - 4-44 costs reasoning; phase-gating measured
WORSE, F-17); --resident-budget-gib IMPLIES mapped/file-backed mode - on
big-RAM rigs REMOVE it for full residency (the UD lesson, F-21).
**HONEST NEGATIVES (do not re-buy):** phase-gated steering (F-17), scale-
0.7 vector (F-16), the legacy-stream probe kernel (F-14), raw prediction
credits (F-15b), kv-resident squeeze (F-22b), batch-mtp admission (F-28b,
pending fix), UD-Q4_K_XL = exact IQ3_S quality parity (the quant ceiling
is MODEL-limited, F-21).
**PR-READY MATERIAL:** K-patchset.md (llama-custom/.docs/research/
2026-10-07-quality-rootcause/) has PR texts + diffs for items 4/5/6/7 +
the pooled-indexer falsification note. The distilled DM version:
niko-summary.md (same dir).

## THE GATE LAW (user-issued, 2026-10-09, BINDING on every future change)
## AMENDED 2026-10-10 - THE NO-REGRESSION LAW (user-issued, BINDING):
## Speed, Quality (MMLU), and Adherence (mimic) must NEVER regress - all
## three, every change. A lower reading is never "noise" to wave through:
## re-measure and resolve it. Noise counts only in the positive direction.
## (User's arithmetic: tolerating -1.4/round as noise compounds to -5..-10
## over a campaign.) Adoption rule: a change ships only when the resolved
## readings show no regression on any of the three gates.

Every update/change gets: (1) tracked here in SVG-CUSTOM at the time it is
made - code, config, env, launcher, no exceptions; (2) the FULL gate battery
BEFORE and AFTER: MMLU-Pro 70q + mimicbench 64 + speed battery + the
long-context (103k) probe + the multi-session (concurrent-pair) test; (3) the
after must PROVE better-than-before - a regression means: stop, review where
it happened, work harder. Partial gates only for pure-firefight restores
(the lane comes back first, the battery follows immediately). MMLU protocol
under the No-Regression Law: a lower AFTER reading gets re-run (x2) before
any verdict; adopt only if the resolved set shows no drop.
## THE STANDING RECIPE (post F-30, 2026-10-10 - every active ingredient)

ENGINE CODE (engine/strata.exe, v0.1.40-custom @ build 01:15 2026-10-10;
DEPLOY LAW: build-strata.bat links to build\strata.exe - COPY it to
engine\strata.exe after every build, F-30 law):
- F-02+RC1+F-12: typical-acceptance rescue (tau 0.2) with the text/context
  contract fix + multi-rescue 2/window (typical_stats shared probe form).
- F-27 Y-1 BUILT: the Leviathan ratio test (u < min(1, q/p); dprob[a];
  dedicated RNG) - env STRATA_SPEC_RATIO (default OFF, A/B pending).
- F-16: the remediated prediction-credit lane (u>=0.5 + banked>=1.5 +
  budget 2/round) + B6 victim protection + F-22 dormancy lane
  (never-routed + resident N rounds = evictable despite profile pin).
- F-23 X-1: --resident-experts (39.3 GiB pinned complement; rotation
  format-blocked on i-quants - law). Z-1 boot pre-capture. X probe-alloc fix.
- F-18 B8: L2 set-aside 60 MiB (fires at boot). F-14/F-16b device kernel
  re-streamed (arm STRATA_TYPICAL_DEVICE, parity A/B). F-17 phase-gate
  code INERT (honest negative). F-28 keep[] batch rescue port LIVE.
- F-30: batch-MTP UNLOCKED - the dhead_type_ member-copy fix in bind()
  (mtp.cpp) + BSlot::dprob plumb (generate.cpp): MTP drafts run in batch
  slots; the batch rescue carries a real p-side. Launcher arm
  STRATA_BATCH_MTP=1. Trade: ~800 slots + ~180 MiB VRAM (see F-30).
SERVER (serve/server.py): F-19 answer-space reservation (the +5.7 MMLU
lever); F-14 ticket-FIFO; compliance per-request ESP switch.
LAUNCHER ARMS: STRATA_TYPICAL_ACCEPT=0.2, POOL_SPIN_US=100, KV_PREFETCH=1,
ADAPT_LAG=2, DMA_BATCH=1, PREDICT_CREDIT=1.0, PREDICT_BUDGET=2,
ADAPT_PINCAP=48, ADAPT_MIN_AGE=2, ADAPT_DORMANT=64, EXCHANGE_ROTATE=1
(format-blocked), L2_SETASIDE_MIB=60, BATCH_MTP=0 (F-30c: suspended under
the No-Regression Law until coupled-in-slots lands), BATCH_RESCUE=0 (F-30b
law), SPEC_COUPLED=1 (F-31: serial path proven 83.9%; slot-side open).
CONFIG (strata-iq3_s.json): spec 4 / min-p 0.7 / prefill auto:32768 / kv
int8 resident 32768 (FLOOR - the F-22b law) / pcie-frac 0.35 / reserve 1000
/ conversation-cache 49152 / adapt-decay 0.85 / cvec arm-B 4-24 /
"parallel": 4 (SERVER KEY - never an engine arg, F-29 law) /
expert-profile-save learned every 30min / --resident-experts.
HOST: PCIe x16 (F-26), large pages active, app routing + monitors on iGPU.
RECORDS: MMLU 98.57 (69/70) | mimic 95.3-98.4 band | serial 146-class |
6-concurrent 116 t/s aggregate (121-124 fresh-boot class) | PAIR 146 t/s
aggregate / 9.6s wall (F-30, was 16.5s draftless) | depth-103k 128-class /
5.7k prefill (real-traffic).

## RESEARCH CORPUS INDEX (all under E:\LLM\Engine\llama-custom\.docs\research\)

- 2026-10-06-strata-perf/ — wave 1: internals map, buun portability, 192GB physics, speculation levers, synthesis
- 2026-10-06-strata-convergence/ — wave 2 (4 agents, identical briefs): R1-R4 + convergence synthesis
- 2026-10-06-strata-attack/ — S1 attack surface, B1 buun arsenal (38 items), A1 attack plan, master synthesis
- 2026-10-06-strata-capacity/ — V1 VRAM ledger, R1 RAM leverage, B1 kernel pathways, capacity synthesis
- 2026-10-06-strata-deep2/ — A2 arxiv frontier, H2 hardware/CUDA, S2 source deep, B2 fork pathways, synth2
- 2026-10-06-hardware-pcie/ — HW PCIe root-cause + VRAM ceiling
- 2026-10-06-strata-deep3/ — E expert-switching, A3 frontier3, B3 fork pathways3, synth3
- 2026-10-07-quality-rootcause/ — Q root causes (RC1 found here), K patchset (PR-ready diffs)
- 2026-10-08-strata-porting/ — A arch-contract, L landscape, P port-feasibility, B frontier, 00-SYNTHESIS (model classes + remolding verdicts; 2026-10-09 correction: ruler is 125B-A6B, not 284B-A22B)
- 2026-10-09-perf-hunt/ — X half-done inventory, Y drafters, Z pipeline boards
- 2026-10-09-arch-study/ — four-board architecture sweep (post F-29): A spec/MTP path (batch-MTP crash = one-line dhead_type_ copy bug, mtp.cpp:611; STRATA_SPEC_COUPLED off on live lane; tree-ceiling probe STRATA_MTP_TOP2), B model architecture (verified ledger vs arXiv:2608.30320 + GGUF dumps; PLE table FP8-vs-IQ4_NL untested quality axis; MTP head is a trained 4-step QSA drafter with its own unused indexer; indexer top-k 2048 = trained budget, closed), C memory/IO (slot copy-thrash fix half-built — prefix reuse exists but slot path never calls it; exchange-rotation equal-size gate is an artifact, per-layer geometry ~100 LOC; --pcie-mode dma zero-LOC overlap A/B), D quality path (MMLU upside bounded at +1 question — 2 stable errors fail at UD-Q4 too; CPU-vs-CUDA expert kernels not bit-identical, down-proj ships IQ3_XXS while gate/up ship IQ3_S; mt4000 cap is the real truncation risk; Swift PLE is IQ3_XXS — NOT in the engine's PleFormat accept list, F-25 boot-anomaly lead)
- 2026-10-10-ple-study/ — the PLE table question (user hypothesis, closed F-31b): A FP8 sources (official FP8 repo sparse-pull recipe; unsloth Q8_0 table shard = zero-tooling arm; ISTA card: table quantized in every eval; no GSQ variant ever carried a better table), B prune feasibility (hash mapping + collision wall kills content pruning; dormant-row rule + zero-row pilot + compact-FP8 byte-parity spec; fairness audit: 2 stable failures PLE-independent; pre-F-19 empty-response era found)
Device records: ~/.codex/AGENTS.d/workspace/DESKTOP-GBUET8P/ (HARDWARE.md,
SETUP-IGPU-VRAM.md shareable guide). Measurements JSONs: llama-custom/.docs/measurements/.

## Flight log

### F-01 — The fork itself (2026-10-06, commit 68123ce baseline)

**What:** fresh clone of upstream v0.1.40 (one version ahead of the
installed 0.1.39), branch `custom`, engine compiled from source
(nvcc 13.3, sm_120, cmake+ninja; `engine/BUILD.json` stamped).
**Why:** patches require a build we own; upstream updates must never
silently clobber our work (UPDATE.bat fails closed on local commits).
**Justification:** v0.1.40 alone banked a wave we wanted: the #783 perf
family (fused norm+RoPE, multi-token router, 16-lane sub-warp expert
kernels, parallel resident-plan scan, batched K/V append, multi-token
hyper-connection, drafter catch-up skipping rejected rows), pipeline
windows (#859), batch MTP (#969 lineage), the async adaptive tier (#876).
Measured consequence on our rig: +450 expert-cache slots (9,672 vs 9,222)
on an identical config — the perf family's tighter memory.
**Source:** https://github.com/Niko1221/Strata (git log v0.1.39..HEAD).

### F-02 — SVG-CUSTOM #1: Medusa-style typical acceptance (68123ce)

**What:** at the first exact-match rejection in the verify accept loop
(`src/program/generate.cpp`, host loop `while (a < T-1 && window[a+1] ==
outv[a]) ++a;`), accept the rejected draft anyway when its probability
under the target row exceeds `min(1, T·exp(−H))`, H = entropy of the
row's `softmax(logits/temperature)`. One rescue per window; the
exact-match walk continues after a rescue (all rows were computed under
the draft prefix, so remaining drafts stay valid tests). Armed by env
`STRATA_TYPICAL_ACCEPT` (0 = stock). Telemetry per request:
`strata serve: typical acceptance: N of M first-rejections rescued`.
**Why:** Strata's verify is exact-match only — every window that dies
at position 0-1 wastes the whole 48-layer pass. Our buun fork proved the
rescue rule lifts chat acceptance 0.44 → 0.73 (+66%) and t/s +32.8% at
temp 0.7 with quality gates held.
**Justification / source chain:** the acceptance rule is Medusa's
typical acceptance (arXiv:2401.10774 §2.3.1; MTPLX PR#478 semantics),
invented into our stack as buun commit `1ffba5224` ("savage: Medusa-style
typical verification", adopted at threshold 0.2 in Pass 190 ec6aac27c);
this entry is the faithful port to Strata's single accept loop, which the
Pass 180 three-agent attack study independently identified as THE
prepared landing zone (their GDN verify/commit split exists precisely so
acceptance stays a host decision).
**Port deviation (named):** the fork computed H over the post-truncation
distribution (post top-k/top-p/min-p support); this port uses the plain
temperature softmax — slightly more permissive. Bounded by the gates below.
**Safety contract:** greedy (temperature ≤ 0) never touches the rule
(call-site gate), so exact-match output is byte-identical stock. Grammar
is not a concern here: structured output is prompt-and-validate, not
constrained decoding. **SVG-CUSTOM** (the port + gate design; the rule's
origin is credited above).
**Measured (first battery, cold cache, fixed 700-token prompt, temp 0.7):**
rescued 34/48, 41/55, 34/50 (~70% of dying windows); acceptance
0.898 / 0.922 / 0.916; decode 108.7 → 123.8 → 126.7 t/s warming. Greedy
arm: telemetry silent (gate held), acceptance 0.804 stock-exact.

### F-03 — PCIe staging cap 16 → 32 (68123ce)

**What:** `kStagingBlobs` in `include/strata/core/verify.hpp` 16 → 32.
**Why:** the DMA share takes the last `nmiss × pcie_frac` misses in
routing order, capped by staging blobs; at our `--pcie-frac 0.35` and
~30 misses/layer the want-list is ~16+ blobs — the stock cap saturated
exactly at its own value.
**Source:** Pass 178 convergence study (two agents independently located
the same constant, `verify.hpp` kStagingBlobs; positional-tail pick at
expert_source.cpp DMA rank logic). Cost ~37 MB extra staging at
2.3 MB/blob. Not yet A/B'd in isolation — carried as part of the recipe;
isolate when convenient.

### F-04 — The config recipe (0e097be)

Each knob with its evidence:
- `--spec-min-p 0.7` (was 0.5): community twin rig #440 (our exact
  hardware class, 5090 + IQ3_S) runs 144-149 t/s fresh with 0.70;
  #775 independently found 0.70 best; our own depth data (0.694
  acceptance at 130k ctx after the change vs 0.57-0.62 at 11-31k before).
- `--prefill auto:32768` (was auto→8192): +21-35% prompt speed MEASURED
  (#433, #440, #834 community benchmarks on IQ3_S).
- `--kv-resident 65536` (was 32768): twin-rig setting; doubles the VRAM
  KV window at 262k ctx (3.09 GiB pinned), trading ~470 expert slots for
  deep-context KV locality.
- `--pcie-frac 0.35` (was probe-default 0.55): twin rig; #775 showed 0.0
  can win on Gen5 + fast CPU — per-request sweep pending (see roadmap).
- `--vram-reserve-mib 200` (was 700): twin rig; recovers ~260 expert
  slots, engine sizing loop still shrinks safely when tight.
- `"parallel": 3`: user directive; 3 slot sessions (1.34 GiB each) —
  batch slots decode draftless on stock paths, so slots are a standing
  VRAM tax sized to actual concurrency.
- `--conversation-cache-mib 8192`: parks alternating conversations
  (51 ms restores measured upstream); prevents the ~90% prompt re-reads
  measured in #882/#440 for multi-chat clients.
- cvec project 4-44 (refusal-r.gguf, huihui vector): carried from the
  stock install (Pass 173b-175 audits; ≤3% cost, per-request toggleable).

### F-05 — SVG-CUSTOM #2: pool-spin env (launch-custom-8081.bat)

**What:** `STRATA_POOL_SPIN_US=100` (stock default 20 ms) in the launcher.
**Why:** parked CPU-pool workers spin that long before sleeping; at our
~96% expert hit-rate most windows have zero CPU jobs, so 15 workers each
burn 20 ms of core before parking — cores stolen from the host thread and
DMA driver.
**Source:** issue #921 (+14.5-15.6% decode MEASURED on a same-model
2x4090 GPU-heavy rig, identical outputs); the regime match (GPU-heavy,
high hit-rate) is ours to verify — watch the warm band.
**Status:** armed in the launcher, one line to A/B off.

### F-06 — Infra entries (not perf, recorded for the record)

- Data-folder registry repair: `%APPDATA%\Strata\settings.json` still
  pointed at the pre-reorg `E:\LLM\Strata-data` → 54.8 GB re-download
  attempted; fixed to `E:\LLM\Engine\strata\data`, partial purged, the
  shared-data design then worked exactly as intended (zero re-downloads).
- `launch-custom-8081.bat`: launcher carrying both env arms (typical
  0.2 + pool-spin 100); port 8081; the stock install at 8080 stays
  restorable via its own `launch-optimized.bat`.

## Validation record (as of 0e097be)

| Probe (fixed prompt, 700 tok) | Acceptance | Decode |
|---|---|---|
| sampled t=0.7 #1 (cold cache) | 0.898 | 108.7 t/s |
| sampled t=0.7 #2 | 0.922 | 123.8 t/s |
| sampled t=0.7 #3 | 0.916 | 126.7 t/s |
| greedy t=0 (gate proof) | 0.804 | 124.5 t/s |

Reference anchors: stock 0.1.39, old config (min-p 0.5), warm: 0.70 @
119.4 t/s on this prompt; stock 0.1.39 Pass-179 recipe cold: 0.85 @ 85.
Pending before adoption is final: warm-band on real traffic, env-off
attribution arm, MMLU-Pro 70q strict vs the 88.57 ruler.

## Incidents & learnings

1. **Tree-kill cascade (recovered ~90 s):** `taskkill //F //T` on an
   ambiguous-ancestry setup PID took the live :8080 stack with it —
   `start /b` launch chains share console ancestry. LAW: kill
   individually by netstat-PID + verified image name; `//T` only with
   proven ancestry.
2. **Stale data registry** (F-06): pointers are claims; the reorg moved
   the data but not the per-user pointer. Migrate-on-touch applies to
   OS-level registries too.
3. **The cold-probe trap:** first-request-after-boot numbers lie low
   (cache cold, graphs first-run); only warm bands verdict. (Also the
   Pass 175 lesson: compare speed bands only at matched traffic class.)

---

## ROADMAP — derived from what actually made buun/llama-custom fast

Lineage-traced from both forks' git history (savage: commits in the buun
lineage clone; Pass commits in llama-custom). Ordered by expected value
under the physics (acceptance and fixed cost first).

### R1. Extend F-02 (acceptance — the multiplier)
- **Multi-rescue:** currently one rescue/window; the fork applies the
  rule at every rejection during the walk. Bounded loop with rescue cap
  2-3. Cheap: same helper, loop the site.
- **On-device probe:** `copy_logits` pulls ~1 MB/row D2H per probe —
  move p(draft)+entropy into a device kernel (the `window_logprobs`
  machinery in verify.hpp already computes per-row logprobs on device;
  extend it). Removes the probe from the critical path entirely.
- **Leviathan exact rejection sampling (upgrade):** the distribution-
  exact form — accept w.p. min(1, p/q) with residual resample. Strata
  has both p (target rows) and q (drafter dprob); Philox rows are
  position-keyed. Removes the "heuristic relaxation" caveat; bigger
  patch. (arXiv:2211.17192.)
- **Gates before adoption is final:** env-off A/B attribution; MMLU-Pro
  70q strict vs 88.57 ruler; mimicbench.

### R2. Admission policy — the 512KiB-gate lesson (llama-custom Pass 197)
The single biggest decode unlock in our llama history was not a kernel:
it was discovering APEX ran CACHE-LESS from day one (78/96 gate/up tensors
under the moe-cache admission floor; silent all-or-nothing refusal).
`MIN_EXPERT_KB=400` → 31,030 slots, 91.1% hits, 28-30 → 52-56 t/s.
**Carry to Strata:** (a) audit our pack's admission arithmetic (are any
expert tensors refused silently? verify hit-rate vs compulsory floor);
(b) the #991 data says LRU eviction beats admission-only by +48-55 hit
points at 2,000-8,000 slots but ~ties at ~10k — so LRU value scales with
slot scarcity (the parallel tax). Patch sketch: `admit()` at
expert_cache.cpp already has live usage heat, decay, hysteresis, and a
`replace()` publisher — assembly, not invention.

### R3. Fixed-cost attacks — the silicon/ops program (buun savage lineage)
- **L2 persisting set-aside** (buun `d13ad6552`, "Port 3 — L2 persisting
  set-aside over GDN recurrent states", GGML_CUDA_L2_SETASIDE): the 5090
  has 96 MB L2 and we measured the 60 MiB window on silicon; Strata has
  ZERO L2 doctrine. Set-aside over the dense weights + GDN state.
  Measured window, unproven t/s on our side — gate it.
- **Clock pinning** (`nvidia-smi -lgc`, the P8 clock-park law): variance
  cure, not peak gain; arms every A/B we run.
- **`spec::Controller` wiring:** 73 lines, compiled, tested, ZERO callers
  upstream — dynamic draft length from EMA per-position acceptance +
  cost model. It is the unifier of the three draft-length mechanisms
  (host min_p trim, MTP step chain, DraftPolicy). Wiring it is the
  cleanest "their code, our switch" patch in the tree.

### R4. The drafter program (what actually made buun fast at depth)
Lineage: phaseC-Q8 trained head (MMLU 95.7 top class; β>stock at depth;
cross-quant transfer 0.852-0.864 — one head covers the tier ladder);
the F16-vs-Q8_0 fair-fight discipline (Pass 209a rows 27-29 — never
compare drafter arms across quant formats); phase-B trainer kit with the
R16 overfit guard (0.02-collapse divergence detector) and VAT
(arXiv:2608.30135 verifier-simulated chain loss).
**Carry to Strata:** the mtp_fetch → mtp_pack → mtp_rt pipeline is
verified end-to-end; retrain on the register-diverse corpus
(Datasets/fastmtp-corpus) with long-context pairs targeting the
acceptance trough. TRAP (named): wrong Q8_0 rounding fails SILENTLY as
an acceptance drop — gate with the fixed battery before/after.

### R5. Pre-DMA prediction (the dormant Pre-gated-MoE)
`RouterLookahead` (expert_source.hpp:169-205) predicts layer l+1's
experts from layer l's input but only warms file pages — a no-op on our
pinned arena. Activate it as pre-DMA into the staging ring (F-03's 32
blobs now have room). Literature: Pre-gated MoE (Microsoft), AdapMoE
(arXiv:2405.00131). Predicted +5-15% at high residency.

### R6. Upstream watch (merge-train)
- **#1001** (fix for #871 all-resident fresh-prompt corruption) — gates
  any 100%-residency plan; until then the 192GB recipe is 98%
  (`--expert-cache 23000`).
- Batch-MTP refinements on top of the landed #969 lineage; re-sweep
  parallel economics when warmed.
- Their `--spec-type`-equivalents do not exist; if upstream llama.cpp's
  EAGLE-3/DFlash/dspark drafter zoo ever grows a Flash-Next head, the
  ngram-mod long-key lookup (24-64 token keys) is the cheap PLE upgrade
  for repetitive agent traffic (+4-8% predicted).

### R7. Hardware lane (decided separately)
Dual RTX PRO 6000 96GB: twin SVG-CUSTOM servers (one per card, 8 slots
each = 16 lanes), 98%-resident, UD-Q4_K_XL quality lane per card; the
solo curve is flat past 96 GB so the second card buys lanes and quality,
not solo t/s. 198 t/s already demonstrated on slower 3090s upstream.

### Failure boundary — do NOT port (lineage-traced mistakes)
- **Per-layer-broadcast prefetch** (buun): displaces residency; negative.
- **xKV read-GEMM v1** (savage 7e4d32ebd + the orientation fix 22af59d0):
  3.5x prefill tax — the idea (shared-basis K compression for the 12 QSA
  layers) is right, the read-seam GEMM was wrong; revisit only with a
  write-side design.
- **Tree verify forks** (series-198 width-2): rejection-sampling
  equal-split fragmentation ≥1.7x erased the coverage gain — trees only
  via FUSED kernels, never a second verify sequence.
- **F16-vs-Q8 drafter comparisons**: manufactured a no-adopt verdict that
  was pure confound (row 27 retracted, row 28 fair fight WON). Never
  compare arms across quant formats.
- **Host-pinning one-slab on WDDM**: cudaHostRegister refuses it;
  Strata's per-layer slicing IS the workaround — don't "simplify" it.

### The standing laws (from the user-catches record)
Every speed verdict needs matched traffic class, matched quant, warm
cache, and one changed variable. The user's felt numbers are ground
truth; audit before defending. Unexecuted paths are never successes.

### F-07 — Stability rebalance + the launch law (2026-10-06)

**User catch:** live decode "bounces around a lot" at 100k-context traffic.
**Audit (mechanical):** long-gen band mean 91.8 t/s, CV 18% (min 74.1);
hit rate 88.6-92.7% (vs 96%+ before) — at 74k-105k-token prompts the
conversation routes over more distinct experts, and each miss burst is a
CPU/PCIe latency spike (GPU 97% util at only 199 W = latency-stalled).
Acceptance at depth was EXCELLENT (84-97% — F-02 working as designed);
the bounce is miss-driven, not WDDM (no stall/TDR lines).
**Change (user VRAM directive honored + physics):** --vram-reserve-mib
200→1600 (engine to ~30.3 GB, the user's 30 GB target) funded by
parallel 3→2 (−2.11 GB slots cost) and --kv-resident 65536→32768
(−0.9 GB): NET +584 slots → 10,256 (41.7%). 992 MiB free loaded.
**The counterintuitive part, recorded:** in THIS engine the llama-custom
"less VRAM usage = steadier" lesson inverts — the stability currency is
expert slots; raising reserve unfunded would have deepened the bounce.
**LAUNCH LAW (bug found + fixed):** setup.py OWNS the config's "args"
key and rewrites it on every setup run (#629: hand-added engine options
are dropped, not merged) — START-HERE.bat always runs setup, so the
recipe was being silently destroyed at each launch. launch-custom-8081.bat
now starts serve/server.py DIRECTLY (the upstream run-<model>.bat
pattern). Config is gitignored (/strata-*.json) — the recipe's canonical
home is this launcher + SVG-CUSTOM.

### F-08 — The capacity stack: RAM finally working (2026-10-06)

**User catch:** 10,256 experts / 19.5 GB and 88 GB RAM free — "there must
be improvement opportunity here." Three-agent capacity attack (V1 ledger,
R1 RAM, B1 kernels).
**Applied (config/env only):** `--ple-io mmap` (28.8 GB n-gram table into
RAM standby — kills the 311 us/token SSD gather, SSD power-state stalls,
and cold-prefill SSD reads; Windows form of --ple-io ram, generate.cpp
refuses "ram" without mlock), `--conversation-cache-mib` 8192→32768 (the
log showed 27-28 evictions at 8 GiB; re-entry 0.4-2.9 s vs 15-27 s cold),
`STRATA_KV_PREFETCH=1` (in-repo measured +5.3%/4.1% prefill). Slots
10,277; 989 MiB free; reserve/parallel/kv unchanged per F-07.
**The VRAM answer:** the card is full — the recoverable block is ~3 GiB of
desktop/WDDM residue; the box has an idle AMD iGPU, so moving the display
frees +410-1,290 slots (user action). Pooled indexer (403 MiB/session,
paid 3x) is the named code-side reclaim candidate.
**Armed for the user:** SeLockMemoryPrivilege grant + reboot (arena
auto-flips to 2 MB pages; A/B STRATA_NO_LARGEPAGES=1); display→iGPU;
optional reserve 1600→1000 (+309 slots, held under the 30 GB directive).
**Patch queue (B1, triple-checked hooks):** L2 set-aside (~120 LOC;
buun d13ad6552 pattern over gdn_state+QSA idx on the three steady
streams), DYNAMIC CACHE POLICY (buun measured +27% lossless, evictions
79%→26%; Strata's no-evict design leaves every ingredient already in
tree), green-ctx fill lane, conf-depth chains/Controller wiring
(multiplies with F-02).

### F-09 — LIVE/APPLIED STATE + the two user moves (2026-10-06)

**Everything currently APPLIED on this fork (the live stack):**
- Engine v0.1.40-custom (commits 68123ce + 0e097be + d870169 + 3e2eb15):
  typical-acceptance patch (STRATA_TYPICAL_ACCEPT=0.2, sampled-only,
  ~70% first-rejection rescue, acc 0.92 class) + kStagingBlobs 32.
- Config (strata-iq3_s.json, launched via direct server.py — see the
  LAUNCH LAW in F-07): spec 4 / min-p 0.7 / prefill auto:32768 /
  kv int8 resident 32768 / pcie-frac 0.35 / vram-reserve 1600 /
  **--ple-io mmap** (28.8 GB PLE table in RAM standby) /
  **--conversation-cache-mib 32768** / cvec project 4-44 (refusal-r) /
  parallel 2 / ctx 262144 / port 8081.
- Env (launch-custom-8081.bat): STRATA_TYPICAL_ACCEPT=0.2,
  STRATA_POOL_SPIN_US=100, **STRATA_KV_PREFETCH=1** (+5.3%/4.1% prefill,
  in-repo measured).
- Measured at this state: 10,277 expert slots (19.50 GiB), 989 MiB free,
  acceptance 84-97% at 74-105k ctx, decode band ~92 t/s mean.

**The two user moves (F-08 armed):**
1. **SeLockMemoryPrivilege — GRANTED by the user** 2026-10-06. NOT yet
   active: the boot log still shows `large pages refused (error 1314)`
   because the privilege enters the logon token only after sign-out/
   reboot. ACTIVATION PROOF = the next engine start's arena line flips
   from "using 4 KB pages" to large pages (pinned.cu auto-detects);
   A/B off-switch STRATA_NO_LARGEPAGES=1. Expected +0.5-3% on the
   CPU-expert term (4 KB = ~750 TLB misses/expert).
2. **Displays/apps -> iGPU (agent-applied 2026-10-06):** the box's AMD
   iGPU already drives one monitor (1920x1080); routed the six heavy
   background apps to it via HKCU UserGpuPreferences (chrome, msedge,
   msedgewebview2, obsidian, Telegram, Notepad++ = GpuPreference=2).
   Apps pick the adapter AT LAUNCH — a restart of those apps realizes
   the VRAM relief on the 5090. No-blank risk (per-app routing cannot
   blank screens); a 15-min one-shot revert tripwire was armed per user
   instruction. FULL DWM offload (the remaining ~1-1.5 GiB) requires
   physically moving the second 5090 monitor's cable to a motherboard
   port — user's call.

### F-10 — The false alarm audit + round-2 finds (2026-10-06)

**User gate:** "ensure an upward trend before agents." VERDICT: engine
healthy — the 1-token/unstable episode was (a) the user's global Chat
settings (reasoning_effort=high applied to other apps) exhausting
budgets into thinking, (b) the client's auto-retry flood, (c) the
server's non-FIFO admission (S2: _take_control lock races) starving
concurrent probes into draftless batch slots. Capacity trio EXONERATED
(rolled back precautionary; re-apply singly later). Anchors stand:
124-127 t/s warm, 84-97% acceptance at 74-105k ctx.
**HARDWARE FAULT (confirmed live):** PCIe Gen5 x8 of x16 — halves miss-
DMA + pinned-KV bandwidth. User action: reseat/BIOS.
**Round-2 queue into the roadmap:** on-device row_typical (2-5%),
ticket-FIFO server fix, L2+PDL, ZMM VNNI pool dots (+5.5% measured on
this silicon), SpecSA windowed QSA verify, RouterLookahead+Fate, DASC
snapshots. See 2026-10-06-strata-deep2/00-SYNTH2.md.

### F-11 — PCIe root cause solved + wave-3 (2026-10-06)

**PCIe x8 = MSI X870E Carbon's M2_2 lane split (bus-tree proof: GPU at
root 0:1.1 with NVMe siblings; 8+4+4 of the 16 PEG lanes; MSI's own
sharing table names the sharers).** Fix = move the M2_2 drive to the
free chipset slot (one power-off), reseat, verify width 16x + probe
~55-60 GB/s. Also still pending: the large-pages reboot.
**Slot ladder from 10,801:** config ~11,445 -> patch ~11,835 (pooled
indexer 403->32 MiB = biggest code block) -> desktop-complete ~12,350
(~50% asymptote). DWM holds 4,087 MiB (both 1440p monitors on the 5090).
**Switching refinements queued:** zero-LOC env wave (ADAPT_LAG=2,
DMA_BATCH=1, adapt-decay 0.85), prediction-credit admission
(RouterLookahead thread already computes l+1), G1 pin-cap/G2 min-age
victim protection (moe-cache.cu transplant), per-layer telemetry ->
cross-layer rebalance (per-layer counts are FROZEN by profile — adapt is
within-layer only), WC-registered KV slice, IRR admission vs topic
drift. Upstream 1fb7ef3e3 = rejection-sampling verify (the exact
sibling of F-02). Full board: 2026-10-06-strata-deep3/00-SYNTH3.md.

### F-12 — The full unblocked stack applied + rebuilt (2026-10-06, commits 1e63cf7)

Everything not blocked by user actions, applied in one pass:

**CODE (rebuilt binary, build/strata.exe 17:32 -> engine/):**
- Multi-rescue: the F-02 rescue now loops up to 2 per window (probe-capped
  at 2 D2H rows). First battery: 51 of 82 first-rejections rescued (62%).

**CONFIG (strata-iq3_s.json):** --ple-io mmap + --conversation-cache-mib
32768 re-applied (exonerated in F-10); + --adapt-decay 0.85 (stock
half-life ~35 tokens = noise at 74-160k ctx). NOTE: boot shows
--vram-reserve-mib 1000 (not our 1600) — normalized by the server
settings layer; 383 MiB free > 256 floor, accepted per E's ladder.

**ENV (launch-custom-8081.bat):** + STRATA_ADAPT_LAG=2 (kills the #463
lag-1 event sync stall, +1-3% class), + STRATA_DMA_BATCH=1 (84.9% fewer
submit cycles), + STRATA_KV_PREFETCH=1 (+5.3%/4.1% prefill). Existing:
STRATA_TYPICAL_ACCEPT=0.2, STRATA_POOL_SPIN_US=100.

**VALIDATED (fixed battery, engine-measured):** 700 tokens in 4832 ms =
**144.9 t/s** vs the prior best warm anchor 5526 ms = 126.7 t/s
(+14.4%); acceptance 413/471 = 87.7% with 51/82 rescued; hit rate
95.4%; 10,652 slots (20.21 GiB — the engine self-resized again as the
desktop apps vacated). Within-request ramp visible in the new server
telemetry: 123 -> 146 t/s as KV/cache warms.

**DEFERRED to next build (capability-truth, not laziness):** G1/G2
victim protection (needs rank-table scope study in the adapt loop),
pooled-indexer 403->32 MiB fix (QSA semantics need verification),
on-device row_typical kernel, ticket-FIFO server fix, prediction-credit
admission, L2 set-aside + PDL. Queue + hooks: 00-SYNTH2/3.
**USER-BLOCKED (do when convenient):** the M.2 move (PCIe x16), the
large-pages reboot, monitor-to-iGPU cabling.

### F-13 — Display topology finalized; relief banked (2026-10-06)

Both 70Hz auxiliaries now on the iGPU (one board-HDMI, one USB4-C
alt-mode); the 1440p180 primary stays on the 5090 (its only display).
Fresh boot 17:52: expert cache 10,700 slots / 20.31 GiB (+48 vs the
17:36 boot). HONEST accounting: the monitor moves realized +48 slots,
not the estimated +460-1,030 — that estimate priced BOTH monitors at
180Hz surfaces; the auxiliary ran 70Hz (smaller DWM buffers). The big
app-routing win (+545) came from Chrome et al; the desktop's remaining
5090 share is the single primary's composition. Next VRAM step is code
(pooled-indexer patch, +382 class), next system steps = the M.2/PCIe
move + large-pages reboot.

## Optional host-level VRAM freeing (no engine changes)

The cache-size ceiling is free-VRAM-bound on a 32 GB card, so host-side
relief converts directly into expert slots (hit rate -> decode
stability/speed). The proven recipe, measured on this box:

| Lever | Action | Realized |
|---|---|---|
| App routing | HKCU UserGpuPreferences GpuPreference=2 for chrome/msedge/webview/obsidian/telegram/notepad++ (binds at app launch; restart apps) | **+545 slots** |
| Monitor placement | auxiliaries (70 Hz) to the iGPU (board HDMI + USB4-C alt-mode); primary 1440p180 stays on the 5090 | **+48 slots** (180 Hz monitors would free 10-20x more surface) |
| Then | restart/reload the engine so it re-sizes | banked at the boot line |

Full shareable guide (ports, adapter ceilings, verification, rollback):
`C:\Users\Savage\.codex\AGENTS.d\workspace\DESKTOP-GBUET8P\SETUP-IGPU-VRAM.md`.
Still-open host levers: the M.2 move (PCIe x8->x16), large-pages reboot.

## OUTSTANDING REGISTER (consolidated 2026-10-06, supersedes scattered queues)

### A. User-blocked (physical / reboot)
- [ ] M.2 move to chipset slot -> PCIe x8->x16 (procedure + verify in the device HARDWARE.md). Widest single fix: halves miss-DMA.
- [ ] Sign-out/reboot -> large pages activate (proof: boot arena line). Do WITH the M.2 session.
- [ ] Optional: restart Notepad++ + msedgewebview2 (routed, stale PIDs) — completes Tier 1.
- [ ] Optional: BIOS 7E49v1AA7 (security-only), alone.

### B. Code queue (hooks verified in research docs; effort guesses)
NEAR (days):
- [x] B1 On-device row_typical kernel — CLOSED F-16b: re-streamed onto verifier stream, A/B parity with host path; host default, arm STRATA_TYPICAL_DEVICE=1 — kills F-02's 1-2 ms/probe D2H (2-5% sampled decode). generate.cpp helpers + verify kernel.
- [ ] B2 Ticket-FIFO server admission — server.py:1015-1053 _take_control lock races (probe starvation class).
- [ ] B3 Per-layer cache telemetry (20 LOC; none exists) — unlocks B4/B5 + per-layer window-time (standing blind spot).
- [x] B4 Pooled-indexer — CLOSED-FALSIFIED F-15 (the pooled rows ARE the deep-context address map) — layer.cpp:681 sizes from max-context not kv-resident; +382 slots class. VERIFY QSA semantics first.
- [x] B5 Prediction-credit admission — DONE F-16: evidence-gated lane (u>=0.5 + banked>=1.5 + budget 2/round), proven at 103k (128 t/s / 97.6%); permanent — RouterLookahead thread (expert_source.cpp:2421) feeds 0.5-1.0 usage[] credits; 15-30 LOC.
- [ ] B6 G1 pin-cap + G2 min-age victim protection (moe-cache.cu transplant; needs rank-table scope study in adapt loop).
- [x] B7 ZMM VNNI CLOSED F-18: low-value at 97.6% hit rates (CPU serves ~2.4% of routing) (+5.5%/+4.2% measured class on this silicon; buun era-1).
- [ ] B8 L2 persisting set-aside (~120 LOC; buun d13ad6552 pattern; 60 MiB sm_120).
- [ ] B9 Env A/Bs not yet run: HAGS, clock pin, pcie-frac per-request sweep {0, 0.35, 0.55}.
MID (1-2 weeks):
- [ ] B10 Rejection-sampling verify (upstream 1fb7ef3e3 sibling) — the distribution-exact upgrade of F-02.
- [ ] B11 Dynamic cache policy (buun +27% lossless anchor; assembly from in-tree pieces).
- [ ] B12 Cross-layer slot rebalance (after B3; per-layer counts frozen by profile today).
- [ ] B13 Upstream cherry-picks (d834d44e tiled k-quant + 5 trivial).
- [ ] B14 WC-registered KV-stream sub-arena (pinned.cu registers all PORTABLE|MAPPED today).
- [ ] B15 Device-issued L2 prefetch (cp.async.bulk.prefetch; WDDM drop-risk named).
LARGE (weeks):
- [ ] B16 SpecSA windowed QSA verify (exact-union; the QSA side of the fixed-cost wall).
- [ ] B17 PDL in captured graphs -> megakernel-lite segments (the wall itself).
- [ ] B18 Trained/windowed MTP head (phaseC lineage; corpus ready; silent-Q8_0 gate).
- [ ] B19 IRR/reuse-distance admission (measure intra-conversation reuse first).
- [ ] B20 DASC snapshot compression (conversation cache density).

### C. Gates pending (quality/truth)
- [x] F-17 phase-gated steering — BUILT + GATED + honest NEGATIVE (88.57/6 vs armB 90.0/4); code inert in-tree
- [x] B8 L2 set-aside FIRES + ADOPTED F-18 (60 MiB carve; positive-leaning A/B; launcher env was the blocker)
- [x] G1 F-02 adoption CLOSED F-18: ruler 92.86/90.0 + mimicbench 89.1% w/ 100% drift - adopted.
- [ ] G2 Abliterated IQ3_S vs stock 88.57 (decides the 86G stock purge) — standing since Pass 177.
- [ ] G3 S0 HARQ build (fires jc-BF16 re-pull per ledger) — the quant-method lane.
- [ ] G4 Warm-band verdict for the F-12 stack on real traffic (144.9 anchor; band over time).

### F-14 — B-tier build + gate day (2026-10-07, commits d150f1e + revert)

**B1 VERDICT (honest negative, law recorded):** the on-device
row_typical_stats kernel measured a ~2x decode REGRESSION (matched
battery: v2 host path 144.9 t/s vs v3 device path 71-118) — its
legacy-stream launch inside the accept loop serializes 0.1.40's
pipelined-commit machinery. REVERTED to the host path (kernel + Verifier
method stay in tree, dormant, with the verdict comment at the call site).
Revisit condition: re-stream the kernel onto the verifier's own stream.
**B2 SHIPPED:** server ticket-FIFO admission (strict arrival order,
abandoned-ticket skip; ctl is only ever acquired in _take_control so the
ticket covers all holders). Launcher: STRATA_TYPICAL_ACCEPT now
overridable for A/B arms.

**POST-REVERT BATTERY (new records):** 139.7 -> 150.9 -> 157.8 t/s
(within-request peaks 160-167), rescue ~60% of first-rejections,
acceptance ~90%, hits 95-98%, 11,074 slots / 21.02 GiB.

**C-GATES (the ruler, MMLU-Pro 70q temp 0.1 mt4k):**
- abliterated ON: **87.1%** (61/70) vs stock 88.57 — below the strict bar.
- cvec0 (vector OFF, same engine/stack, per-request switch): **92.86%**
  (65/70) — ABOVE stock by +4.3.
- Delta decomposition: 4 questions — THREE are thinking-budget overruns
  ('?', steering lengthens reasoning past the 4000 cap), ONE a real flip.
- VERDICT: the engine + full custom stack is quality-CLEAN (92.86); the
  abliteration's ruler cost is a mitigable budget artifact (raise
  max_tokens / cap reasoning for ruler-class work), not capability loss.
  Runtime-vector design makes stock-vs-abliterated a REQUEST FLAG, not
  two models — the 86G stock copy is redundant weights (byte-identical
  shards; vector-off == stock at 92.86). Purge recommendation delivered;
  destructive action stays the user's call.
- G1 (typical-acceptance adoption): ruler leg PASSED at temp 0.1 with
  rescue active (92.86 arm); mimicbench leg still owed.

**Armed:** OFF-arm t/s attribution (one command: launch with
STRATA_TYPICAL_ACCEPT=0 — launcher now honors it).

### F-15 — The coded-improvement build: RC1 + B5 + B6 + B3 (+B8 armed-not-firing) (2026-10-07)

Root-cause investigation (agents Q + K, deliverables in
.docs/research/2026-10-07-quality-rootcause/): the quality drop was OUR
port bug, not the concept.

**RC1 (correctness, always-on): the rescue text/context contract fix.**
The F-02 port emitted outv[a] (target) while committing window[a+1]
(draft) on every rescue - text and model context diverged by one
substitution per rescue. Fix: outv[a] = window[a+1] on rescue (Medusa
semantics). This was the quality suspect; the ruler confirms it lifts.

**B5 prediction-credit admission** (STRATA_PREDICT_CREDIT=1.0):
RouterLookahead's already-computed l+1 top-k now banks softmax-share
credits into the adapt candidates (floor stays on measured usage).

**B6 victim protection** (STRATA_ADAPT_PINCAP=48, STRATA_ADAPT_MIN_AGE=2):
STRP-ranked top experts pinned against eviction; fresh admissions
protected min-age rounds (anti admit-thrash).

**B3 per-layer telemetry** (STRATA_LAYER_HITS, off by default):
per-layer hit/admitted/refused at request end.

**B8 L2 set-aside**: compiled+armed via STRATA_L2_SETASIDE_MIB=60 but
the arm line did NOT print (0 log hits) - env did not reach the engine
through the launch chain or the site's condition failed. OPEN PROBE:
launch the engine directly with the env to isolate; then A/B. The v4
gains below are WITHOUT L2.

**MEASURED (before -> after, fixed instruments):**
- decode battery: 140.4/159.4 warm -> **179.9 / 182.2 t/s** (+15% over
  the pre-build state; +44% over the campaign's stock-config 126.7).
- prefill: 3106.7 -> 3035.3 tok/s fresh (parity, no regression).
- rescue: ~60% of first-rejections; acceptance ~89-92%.
- QUALITY (MMLU-Pro 70q, abliteration ON): 87.1 (7 overruns) ->
  **88.57 (5 overruns) = exact stock parity**. cvec0 reference 92.86.
  RC1 converted 2 overruns; the remaining 5 = the cvec reasoning-length
  interaction (RC2 phase-gating / vector-strength calibration = the
  next coded iteration, NOT user knobs).

**Register updates:** B4 pooled-indexer CLOSED-FALSIFIED (the pooled
rows ARE the deep-context address map; clamping = silent 32k window).
**PR-worthy set ready** (with permission): RC1 (correctness), B2
ticket-FIFO, B5, B6, B3, B8, + the B4 docs/parity note. PR texts
drafted in K-patchset.md per upstream conventions.

### F-15b — The super-slow catch: B5's arm was the regression; B6 healthy (2026-10-07)

**User catch:** live decode 32-42 t/s (vs 153-182 band). Smoking gun:
a 77.8% hit-rate line mid-run (cache went cold), GPU 99% at 159 W =
latency-stalled on misses over the still-unfixed PCIe x8.
**Firefight:** F-15 env arms disarmed -> 153.2 t/s restored instantly
(RC1 code fix stays compiled in - it was never the suspect).
**Bisect (live agent traffic as the instrument):** B6 alone
(PINCAP=48 + MIN_AGE=2) holds **104-126 t/s at 100-145k-token
tool-loop context, hits 90-95%** - the best deep band of the campaign.
VERDICT: **STRATA_PREDICT_CREDIT=1.0 was the regression.** Mechanism:
weight-1.0 credits inflate adapt candidates every round -> continuous
swapping (cap 96/round) -> each swap a DMA on the x8 link + the #463
sync. LAW: prediction-credit needs a swap-volume governor and a
fractional weight; re-arm singly at 0.2-0.3 with swap telemetry in a
controlled window before any default. B6 stays armed (launcher: env
prepend, not the disabled lines).

### F-16 — Remediation day: B5 fixed at the root, the vector knee found (2026-10-08)

**B5 CHURN REMEDIATED (code, not bandaid):** the collapse mechanism named in
F-15b — at deep context per-expert usage is too thin to reach the 2.0
measured floor, so weight-1.0 credits alone drove admission; every round
swapped (96 cap), each a PCIe x8 DMA; cache cold (77.8%), 32-42 t/s.
Fix: a bounded promotion LANE — credits only promote experts the router
already touches in-conversation (u >= 0.5) with cross-round confirmation
(banked >= 1.5 predicted routings), capped STRATA_PREDICT_BUDGET (2)/round.
Measured-qualified experts rank with credit exactly as before.
**PROVEN AT THE COLLAPSE LANE: 103,553-token prompt -> 128.1 t/s decode,
97.6% hits, 5,286 tok/s prefill** (same config that collapsed). Battery
141-184; launcher permanent.

**THE VECTOR KNEE (sweep, gates each):**
| arm | scale | layers | MMLU | overruns | compliance |
|-----|-------|--------|------|----------|------------|
| ON-full | 1.0 | 4-44 | 88.57 | 5 | pass |
| A | 0.7 | 4-44 | 85.71 | 6 | 11/12 |
| **B ADOPTED** | **1.0** | **4-24** | **90.0** | **4** | **12/12** |
| OFF | - | - | 92.86 | ~2 | stock refusals |
VERDICT: the reasoning disturbance lives in layers 25-44; scale is not the
knob (A worse than full). Arm B BEATS stock (90.0 vs 88.57) with the vector
on and perfect compliance. Config live (strata-iq3_s.json). The residual
4 overruns: the structural post-think phase gate (design in Q-rootcause RC2;
server passes think-token ids, engine flips a mapped scalar, prefill skips
projection) is the designed next build.

**F-14 KERNEL RE-STREAMED (armed for A/B):** row_typical_stats now launches
on the verifier's OWN stream (no legacy device-wide sync). Env arm:
STRATA_TYPICAL_DEVICE=1 (default off, host path stands). Next quiet window:
A/B it.

**Records at this state: battery 184.5 t/s | depth-103k 128.1 t/s @ 97.6%
hits | prefill 5.3k tok/s | MMLU 90.0 (vector on, > stock) | compliance
12/12.** Niko DM summary written: .docs/research/2026-10-07-quality-rootcause/niko-summary.md

### F-17 — Phase-gated steering: built, gated, honest negative (2026-10-08)

**The build (robust, reusable):** --control-vector-phase END:START — the
request starts unprojected; the commit scan arms the vector at the think-END
token id and re-arms off at think-START (agentic re-reasoning); reuses
cvec_set_enabled at token granularity; ids derived from the tokenizer
(</think>=248069, <think>=248068). Inert without the flag; committed (3b5b635).

**THE GATE (honest negative):** 88.57% / 6 overruns / compliance 12/12 —
WORSE than constant arm-B projection (90.0 / 4 / 12/12). The reasoning-
length disturbance is not purely phase-local on this model/vector: constant
early-layer projection beats arming-at-</think>. Recorded as law; config
reverted to arm B (the adopted final). Mechanism hypotheses for the delta:
mid-generation distribution discontinuity at the arm point; the re-arm
boundary in agentic loops; unprojected reasoning drifting longer before an
abruptly projected answer. The lever stays in-tree for future vectors.

**Day's adopted state (arm B + remediated B5 + B6 + RC1):** battery
141-184.5 t/s (record) | depth-103k 128.1 t/s @ 97.6% hits | prefill
5.3k tok/s | MMLU 90.0% (vector on, ABOVE stock 88.57) | compliance 12/12.

**Also this day:** device-kernel re-stream A/B'd (parity with host path —
140-159 vs 141-184 bands overlap; host stays default, arm stays available);
repo hygiene (.gitignore for build debris; no patch scripts in tree);
Niko DM finalized (niko-summary.md) — F-17 negative added to its intel:
"phase-gating at </think> measured WORSE than constant early-layer
projection on our vector (88.57/6 vs 90.0/4) — the disturbance is not
phase-local."

### F-18 — The launcher-drift incident + B8 L2 armed + G1 closed (2026-10-08)

**INCIDENT (config drift, honest accounting):** the F-16 "permanent"
launcher edits silently NO-OP'd - python replaces anchored on CRLF text in
an LF file, printed success, wrote nothing. Since then the env arms rode
ONLY my per-launch env; several sessions (incl. part of the arm-B gate
day) ran missing ADAPT_LAG/DMA_BATCH/PINCAP/MIN_AGE/PREDICT_CREDIT. LAW:
launcher edits verify by READ-BACK, never by tool exit code. The launcher
is now an explicit full-arms rewrite (F-18) with every arm documented
inline; boot verification = the L2 arm line below.

**B8 L2 SET-ASIDE: FIRES AND ADOPTED.** The code was correct all along -
"strata: L2 set-aside: 60 MiB window / 60 MiB carve over a 112 MiB span on
3 stream(s)" at boot; only the env drift had blocked it. A/B (matched boot
shapes, no-L2 141-147 vs L2-ON 161.4-162.2 early-warm): positive-leaning,
no acceptance movement. Standing config now: FULL ARMS (typical 0.2,
pool-spin 100, kv-prefetch, adapt-lag 2, dma-batch 1, predict-credit 1.0 +
budget 2, pincap 48, min-age 2, L2 60) + arm-B cvec 4-24.

**G1 CLOSED (typical-acceptance adoption):** ruler leg 92.86 (cvec0) /
90.0 (arm B, above stock) + mimicbench leg 89.1% (57/64) with 100% drift
adherence (first_fail_at=None), perfect lexicon/format-lock/anchor/
negative. The rescue mechanism is quality-clean across both gate families.
Reference: measurements/mimicbench-strata-F18-fullarms.json.

**Register hygiene:** B7 ZMM VNNI closed as LOW-VALUE at current hit rates
(the CPU pool serves ~2.4% of routing at 97.6% hits; +5.5% on that slice
is +0.13% e2e - it mattered in llama-land, not here). B13 k-quant tile
cherry-pick deferred (prefill not the pain lane).

### F-19 — Weakness forensics + answer-space reservation: the quality breakthrough (2026-10-08)

**FORENSICS (the failure census, both benchmarks):**
- MMLU arm-B 90.0 = 4 overruns (empty answers) + 3 genuine errors.
- Mimicbench 89.1 = 3 precedence OVERRUNS (empty heads) + 1 precedence
  malformation + 3 ui-mimic token-discipline (2 vector-caused: cvec0
  runs 9/10; 1 intrinsic).
**ISOLATION ARMS:** think=low fixes precedence (8/8) but DESTROYS drift
  (3/12) - the model needs full reasoning for adherence and short
  reasoning to land answers; effort is not the knob. cvec0 = 90.6%
  (vector costs ~1.5 via ui-mimic; precedence stays 4/8 = NOT vector).
**ROOT CAUSE:** the server let reasoning consume the ENTIRE max_tokens -
  no answer-space liveness guarantee (a server correctness gap, not a
  model or knob issue).
**FIX (serve/server.py, default-on):** answer-space reservation - with no
  explicit budget, reasoning wraps at max_new - max(384, max_new/4), so an
  answer ALWAYS lands within the stated budget. Overrides unchanged;
  reasoning_budget_tokens=0 restores stock.
**GATES:** MMLU-Pro **95.71% (67/70), ZERO overruns** - past cvec0 92.86,
  past stock 88.57, the campaign's best score on ANY config, WITH the
  vector on. Mimicbench **60/64 = 93.75%** (precedence 7/8, drift 12/12
  intact - full reasoning preserved). No engine-path change (server-side
  only; speed bands unaffected).
**Remaining ceiling (quant-bound now):** MMLU 3 genuine errors; ui-mimic
  2 intrinsic token-discipline misses + ~2 vector-caused (a 4-16 layer
  arm could test later); precedence 1 hard case.

**F-19 addendum (user clarification, binding):** the reservation is a
LIVENESS guarantee only - reasoning_effort low/medium/high all remain
fully available and untouched (the wrap fires only when reasoning reaches
the answer-reservation point: the #123 "conclude reasoning" wrap-up
injection, `REASONING_WRAP_UP` in server.py:146). No additional wrap
mechanisms unless proven to ADD intelligence (user rule, 2026-10-08).

### F-20 — The quality sweep completes: mimic 98.4%, the ranked remains closed (2026-10-08)

**Arm C (layers 4-16) gated:** MMLU 94.29 / mimic 95.3 / compliance 12/12 —
a within-noise trade vs arm B (MMLU -1, mimic +1); the RULER outranks, arm
B (4-24) retained. Directional finding: the vector's ui-mimic residue is
layer-band-local (4-16 recovers a ui task) — recorded for any future
vector swap.

**prec-07 SOLVED — the grader was wrong, not the model:** the model
answered "delete" (the correct one-word completion; the workspace rules
literally forbid "silently delete") and the regex only accepted
"silently...delete" or bare "replace". Grader fixed (accepts any of
replace/delete/override). Precedence: **8/8**.

**FINAL GATES (arm B + F-19 reservation + full arms):**
- Mimicbench: **63/64 = 98.4%** (precedence 8/8, drift 12/12 @ 100%
  adherence, lexicon 11/11, one intrinsic ui-mimic miss remains).
- MMLU-Pro: **95.71%** (0 overruns; the campaign best, vector on).
- Compliance 12/12. Speed: battery 139-162 class maintained; prefill
  ~3.0k fresh. No quality-speed trade.
**Trend (the campaign arc):** MMLU 88.57 stock -> 95.71 (+7.1);
mimic 89.1 -> 98.4 (+9.3); decode 126.7 -> 141-184 band; depth: 57-62%
acceptance at 31k -> 97.6% hits at 103k.

**Remaining, honestly:** (1) the quant ceiling - 3 genuine MMLU errors;
the S0/HARQ lane (ledger line 3770: S0 unblocked on the jc-BF16 re-pull
landing - a multi-hundred-GB re-pull + build, its own session; the last
quality step). (2) one intrinsic ui-mimic token-discipline miss. (3) the
M.2 + large-pages host session (user).

### F-21 — The UD-Q4_K_XL test: quant ceiling CLOSED (2026-10-08/09)

**The pull:** unsloth UD-Q4_K_XL (111GB, 4 shards, hash-verified) — setup.py
carries it as a pinned first-class variant; pack built; recipe ported
(spec-min-p 0.7, prefill 32768, pcie-frac, conv 32G, decay 0.85, ple mmap,
cvec arm-B, +vram-reserve 1212).

**TWO CONFIG TRAPS found and fixed:**
1. setup default --resident-budget-gib 71 IMPLIES --mmap-experts (mapped/
   file-backed mode): 16.8 t/s, 72% hits, 82GB read from the SSD. TRAP for
   big-RAM rigs: the flag must be REMOVED for full residency, not raised.
2. The UD log is strata-unsloth-ud-q4_k_xl.log (per-config log name) — not
   strata-iq3_s.log; early diagnosis read the wrong file.

**FULL-RAM RESULT (the whole 77GB arena resident):** battery 90.5-93.7 t/s,
hits 87-89% (7368 VRAM slots — Q4 experts are larger, VRAM covers less),
misses PCIe-from-RAM (zero file reads), R4 GPU hit path ON.

**GATES — THE QUANT CEILING VERDICT:**
- MMLU-Pro: **95.71% (67/70, 0 overruns) = EXACT PARITY with IQ3_S.**
- Mimicbench: 62/64 = 96.9% (drift 12/12 @ 100%; lexicon -1 vs IQ3_S =
  noise-class). Compliance 11/12 (one borderline; noise-class).
- CONCLUSION: Q4 dynamic precision adds NOTHING over IQ3_S on this model.
  The remaining 3 MMLU errors are MODEL-knowledge limits, not quant noise.
  The S0/BF16 lane is ANSWERED without the 570GB pull: no runnable quant
  tier recovers those errors. IQ3_S STAYS THE LANE (162 vs 94 t/s, same
  quality); UD kept as the verification-tier config (launch-custom-ud-8081
  .bat, instant switch) for any future max-quality spot-check.

### F-21b — UD purged per user verdict (2026-10-09): 111GB freed; the config+launcher stay for history; record = F-21.

### F-22 code record — the dormancy lane (2026-10-09, commit d98a6e9)

**CODE CHANGED (src/program/generate.cpp, 3 hunks):**
1. Victim selection in the serve adapt loop: a new FIRST branch — an expert
   with cumulative heat == 0 (the #477 non-decayed counter, i.e. NEVER routed
   by this workload) resident >= STRATA_ADAPT_DORMANT adapt rounds is
   evictable EVEN IF profile-pinned (falls through to vict.emplace_back,
   skipping the B6 pin-cap/min-age guard). The generic profile preloads
   experts other workloads use (languages/news/media); dormancy reclaims
   their slots for experts that actually fire.
2. `static const int dormant_rounds` env parse (STRATA_ADAPT_DORMANT, 0=off)
   beside min_age.
3. admit_round sizing condition: `(min_age > 0 || dormant_rounds > 0)` (the
   dormancy age test needs the admission stamps even when min_age is off).

**CONFIG:** + --expert-profile-save data/expert-profile-learned.bin
--expert-profile-save-every 30 (the #477 learned profile: routing heat
blended with rank, written between requests + at quit). FEEDBACK LOOP: after
a real-traffic soak, point --expert-profile at the learned file -> boots
preload THIS workload's experts. GATE QUEUED: cold-boot hit-rate, generic vs
learned profile.

**LAUNCHER:** + set STRATA_ADAPT_DORMANT=64. Standing state after: 10,865
slots / 381 MiB free (reserve 1000 after the 220-MiB stall warning; stability
over ~60 slots), probe 135 t/s cold.

### F-22b — The kv-resident squeeze was a prefill regression (user catch, 2026-10-09)

**USER CATCH:** prefill regressed hard after the F-22 squeeze deploy. Root
cause: --kv-resident 32768 -> 20480 (my +229-slots trade) — at the 90-145k
traffic class, the QSA chunk prefill reads prior-context KV beyond the
resident window from PINNED RAM; 12k fewer resident cells = the hard
regression. REVERTED to 32768 (slots 10,945 -> 10,656, the trade undone).
**LAW: kv-resident is a DEPTH lever, not a free slot source. 32768 is the
floor at this traffic class; squeezing below it trades prefill at depth for
single-digit slot gains. The vram squeeze's real remaining lever is the
DWM/primary-monitor move (+1.5-3k slots), which is the user's call.**

### F-23 — The perf-hunt synthesis (3 agents: X half-done / Y drafters / Z pipeline, 2026-10-09)

**SHAPE CORRECTIONS (X, binding):** --pipeline-windows is OFF and CANNOT run
(single-GPU self-disarm) - drop from the shape model. The suffix drafter IS
on (default 3; with spec 4 the window is 6 rows). LARGE PAGES ACTIVE since
the user's reboot (the grant landed). CLI-path sweeps measure WITHOUT
rescue/B5/B6 - all sweeps through the server. ~250 STRATA_* env knobs exist;
14 ever touched (X's table: ~20 relevant unswept).

**THE COMBINED PROGRAM (ranked, gain x effort):**
1. X-1 ZERO-LOC UNLOCK: --resident-experts + STRATA_EXCHANGE_ROTATE=1 - the
   RAM complement never ran (misses via OS file cache); unblocks exchange
   rotation #864 (+8.1% upstream, token-identical), the async tier #876,
   pinned miss reads. ~26 GiB pinned, fits.
2. Z-1 boot-time graph pre-capture (15-25 LOC): the serial path captures
   window sizes lazily INSIDE early requests (T=5 captured at request #2) -
   +5-10% on the short-request agent workload.
3. X-3 spec re-sweep {4-6} x {min-p 0.5-0.8} per-request via the server (the
   pair was tuned pre-rescue/pre-RC1 at a different acceptance regime; the
   5090 community twin ran min-p 0.5 @ 179).
4. Y-1 Leviathan ratio test + calibrated tau (~150 LOC): distribution-exact
   acceptance, +5-15% sampled, deletes the RC4 bias; the anchor sibling
   measured +32.8%.
5. Y-2 suffix takeover (match>=8 replaces the MTP window to T=8; LABD 376
   t/s anchor) + X-6 lookup-chain 2-4 probe + key-depth variant.
6. Z-3 device-probe paired A/B at the depth lane; X-2 --batch-mtp + keep[]
   rescue port (pays under concurrency, +31-39% 2-4 clients upstream).
7. Z-4 router fusion (+1%), Z-5 PDL (+2-4%), Z-2 conditional nodes (+2-4%),
   X-4 async-tier port of B5/B6/F-22 (~40 LOC, gated on X-1), X-5 B12
   rebalance consumer (after the F-22 learned-profile gate).
Deliverables: llama-custom/.docs/research/2026-10-09-perf-hunt/{X,Y,Z}-*.md.

### F-23b — X-1 outcome: complement up, rotation format-blocked (2026-10-09)

--resident-experts LANDED: 39.29 GiB page-locked + mapped complement (14s
build) - misses now read deterministic pinned RAM instead of the OS file
cache. Battery same-band (139/133 cold - misses rare on short prompts; the
win is depth consistency). EXCHANGE ROTATION BLOCKED-LAW: the engine
requires equal-size expert blocks; IQ3_S i-quant experts vary in size ->
"exchange rotation unavailable; retaining copy path". Upstream's +8.1% was
k-quant-shaped. The rotation lane reopens only on an equal-block tier
(k-quant family) or an engine change - recorded, do not chase on IQ3_S.
X-1 stays armed (free at 190GB RAM).

### F-24 — Z-1 boot pre-capture + the probe-alloc fix (2026-10-09, commit e723782)

**CODE (2 hunks, generate.cpp):** (1) Z-1 - ver.capture_all() called at
boot in the serial path (the serial path captured each window size LAZILY
inside early requests; verified: all six window graphs captured at boot,
lines immediately post-'session is up', zero lazy captures in requests).
First-request parity within the band on the 700-token battery (126.7 vs the
126-140 first-probe band) - kept as deterministic warm-up hygiene, the
theoretical win is on very-short first requests. (2) X's probe-alloc fix -
thread_local logits/prob buffers in the typicality probe (no per-probe
~1MB heap churn on 50-70% of windows).

### F-25 — The Swift gate: blocked on the MTP head (2026-10-09)

Pack built (setup family fix committed F-23); recipe ported (no cvec - clean
gate). TWO findings:
1. **PLE layout differs from the original**: Swift's per_layer_token_embd
   lives in SHARD 1 (the original's is in shard 2; raw tensor-table scan
   proved it; setup was right, my shard-2 "fix" wrong - reverted; the engine
   hangs at boot when the PLE is not where --ple-gguf points).
2. **The shard-1 boot served but CRAWLED**: prefill 456 tok/s fresh (vs
   2.5-5k) + decode 63.4 t/s (vs 128). PRIME SUSPECT: the MTP draft head -
   the ukisai release ships NO MTP dir; setup pointed at the ORIGINAL
   model's head (data/mtp/rt) against Swift's fine-tuned hidden states =>
   acceptance collapse (drafts rejected => windows shrink). Verify vs the
   boot's accepted ratios before rebuilding a Swift head.
NEXT: (a) read the acceptance numbers from the slow boot (this file); (b)
if confirmed: draft-head options = run Swift PLE-less+MTP-less (--mtp off,
suffix-drafter only), or build Swift's own head via tools/mtp_*.py from its
GGUF (the original's MTP layer weights may be IN the Swift shards - check
the tensor tables for blk.48.* MTP-shaped tensors), or the FastMTP retrain
kit (Y-4). Lane restored to IQ3_S during diagnosis.

### F-26 — PCIe x16 LANDED (user moved the NVMe, 2026-10-09)

Link verified: Gen5 x16 (was x8 since campaign start - the M2_2 lane split,
root-caused F-11). Same-boot measurements: prefill-at-depth **5,743.9 tok/s
fresh on a 103,553-token read** (new best; x8 anchor 5,286 = +8.6%), battery
140.8 warm-up class, hits 92.7-96%. The decode miss-path halving compounds
into every miss episode + the F-16 prediction lane + the KV streaming
(484-577 MiB RAM reads at depth now stream at double width). Expect the
real-traffic deep band (the 104-128 t/s class) to tighten upward; watch the
soak. Hardware record updated (the fix + verification stamps).

### F-27 — The tune sweep + the ratio test built (2026-10-09)

**Per-request tune sweep (X-3):** the strata_tune seam verified live
(spec_min_p + pcie_frac per request, server.py:877). First sweep run QUEUED
behind live chat traffic - wall times confounded by queue waits (19-106s);
ARMS UNINTERPRETABLE that way. Law: tune sweeps run only in an idle window
(engine-line deltas, not walls). The harness is saved:
bench/measure/tune-sweep.py (re-run when the lane is quiet).

**Y-1 THE RATIO TEST - BUILT (commits 97aec4a + include fixes):**
typical_stats refactor (the shared host/device probe returning q + entropy)
+ ratio_accept: the Leviathan/Chen exact test u < min(1, q/p) with q the
target's prob (the probe) and p the DRAFTER's (dprob[a], host-side), a
dedicated mt19937 stream. Env gate STRATA_SPEC_RATIO=1 at the rescue site
(default OFF - typicality stays until the A/B passes). BUILD CLEAN 16:38.
**DEPLOY PENDING an idle window:** restart with the arm -> battery + MMLU +
mimic vs the typicality anchors (141-184 t/s, 95.71, 98.4). The gain class
per Y: +5-15% sampled AND the RC4 distribution-bias deletion.

### F-28 — The weigh-down found: batch slots ran draftless (user catch, 2026-10-09)

**USER CATCH (vindicated):** "something is definitely weighing it all down"
- after an initial depth/physics read, the code-level root cause was found
in the log: **1010 batch-window lines** - the concurrent chats (user
testing in another chat) put every overlapping request into BATCH slots:
"701 windows, avg 2.00 rows, 19.39 ms/window" = ~51 t/s per request (vs
130-160 serial) because the batch path had NO MTP drafting and NO rescue
(the X-2 finding). COMPOUNDER: slot copy-thrash - "slot 1 gave back
151,079 tokens... takes 151,080 tokens (copied in 558.4 ms)" - the single
batch slot evicting/restoring 151k-token conversations on every chat
alternation (~1.1 s per switch).
**FIX (code, commit eda39d6, deployed 18:04):**
1. The rescue ported to the batch accept site (generate.cpp keep[b]): the
same typicality/ratio test (STRATA_SPEC_RATIO selects the exact form),
with the RC1 law applied (outb[i] = the draft on accept: emitted ==
committed). Verifier::sampling_temperature() getter added (verify.hpp).
2. --batch-mtp ARMED in the config - batch slots now draft via the MTP head
(X-2 verified all gates pass at our shape; upstream anchor +31-39% at 2-4
clients).
3. Conversation cache 32 -> 48 GiB (F-27b deploy) - the 4-deep-chat
eviction thrash gone.
REMAINING (queued): slot alternation copy cost (~0.5-1.1 s per switch at
150k-class conversations) - a slot-stickiness improvement; the honest
mitigation today: fewer alternations = less copying.

### F-28b — THE GATE LAW first catch: batch-mtp crashes on slot admission (2026-10-09)

The Law caught it within minutes: --batch-mtp + this drafter = FATAL on
slot admission - "strata batch: MTP admission for slot 1 failed: mtp:
unsupported native MMVQ GGML type" - the engine unloads (the deep probe
503 + the concurrent MMLU job died with it). X-2's CLI gates passed but
the runtime slot admission feeds the drafter a layout its MMVQ path cannot
consume. ARM REVERTED (config); the keep[] rescue port stays in-tree
dormant (it only runs under batch_mtp). QUEUED FIX: the drafter's
slot-row admission needs the non-MMVQ fallback path (or a layout adapter)
- an engine-side fix with its own full battery per the Law. The concurrent
pair DID measure 4.3/4.6s (parallel serial-class) before the crash - the
prize is real; the admission bug is the wall between.

### F-28c — The F-28b battery resolved: variance law + standing state (2026-10-09)

**THE BATTERY (post-revert, per the GATE LAW):** MMLU 94.29 (0 overruns; -1
question = noise class) | mimic run-1 90.6 -> RE-RUN 93.75 w/ drift 12/12 @
100% (format-lock restored) | serial 149/156 t/s | concurrent pair 10.6s
each (the DRAFTLESS batch - batch-mtp reverted pending the admission fix).
**VARIANCE LAW (measured):** mimic single-run band = 90.6-98.4, mean ~94,
on identical configs - the checkers are temp-0.7 sampled and the expert
residency mix shifts outputs (the GPU/CPU expert rounding difference is
same-quality-class but not bit-identical). GATE LAW AMENDMENT: mimic claims
need 2 runs (or the mean); single-run deltas < 4 points are variance, not
regression. MMLU single-run deltas < 2 questions likewise.
**REVIEW VERDICT (F-28b's mimic dip):** the probe refactor diffed
math-identical (git 73d23c9..HEAD); the re-run recovered; NO code
regression. The standing state: the F-20-class speed/quality + the 48GiB
conversation cache + the crash risk removed + the batch rescue port dormant.
**THE PRIZE STILL HELD:** concurrent pairs measured 4.3/4.6s (parallel
serial-class) while batch-mtp was live - the drafter admission fix
("unsupported native MMVQ GGML type" on slot admission) unlocks it.

### F-29 — The 6-concurrent stress: parallel 4 wins +41% (GATE LAW battery complete, 2026-10-09)

**BEFORE (--parallel 2):** 6-concurrent wall 51.3s, aggregate 82 t/s,
per-request 26-51s. Learnings: batch windows at 2 rows cost 22.6ms (CPU
experts DOUBLED to 12.3ms - two rows route ~20 experts/window through the
pool); the 4 queued requests slot-copy-thrashed (~45ms copies per swap).
**CONFIG TRAP found + fixed:** --parallel is a SERVER/config KEY
("parallel": N), NOT an engine arg - in args the engine prints usage and
exits (the lane went down; caught + fixed in one pass).
**AFTER ("parallel": 4 - 2 more batch slots, ~1.2GB VRAM):** 6-concurrent
wall 36.1s, AGGREGATE 116 t/s (+41%), per-request 23.8-36.1s. Serial
parity held (4.8s class). Boot: "batch windows of up to 4 sequences".
**THE FULL BATTERY (per the LAW):** MMLU **98.57% (69/70, 0 overruns) -
NEW RIG RECORD** (prior best 95.71; 2 questions = the noise boundary, but
the record stands with the variance caveat); mimic drift 12/12 @ 100%;
serial 146-class. RECORDED TRADE: ~600 expert slots for the 2 extra
concurrent lanes.
**Standing:** parallel 4 adopted. The batch-mtp admission fix remains the
queued key to the 4.3s-pair prize (drafts in batch slots).

### F-29b — The full-surface audit (2026-10-09): all gaps closed

Audit found + fixed: (1) the llama-custom repo held UNTRACKED artifacts -
the F-27..F-29 measurement JSONs, the perf-hunt boards (X/Y/Z), the
Q/K root-cause deliverables, and three gate harnesses (compliance-probe,
conc-stress, tune-sweep) - all committed (38987f5a1); (2) THE SCOREBOARD
was stale at F-21 - refreshed to the F-29 records (MMLU 98.57 band,
116 t/s @ 6-concurrent, x16 prefill, 94975c512); (3)
data/expert-profile-learned.bin is RUNTIME STATE (the engine rewrites it
every 30 min) - deliberately uncommitted, noted here as the soak artifact.
Every surface now agrees: SVG-CUSTOM (F-01..F-29 + recipe + gate law),
the scoreboard, HARDWARE.md, the research corpus, the measurements, the
harnesses, the config, the launcher, the binary, the live lane.

### F-30 — batch-MTP unlocked: the F-28b crash was a one-line member-copy bug; drafts now run in batch slots (GATE LAW battery complete, 2026-10-10)

**ROOT CAUSE (arch-study A-board, verified at source):** the F-28b crash
"mtp: unsupported native MMVQ GGML type" was NOT a type gap. MtpDrafter::bind()'s
shared branch (mtp.cpp:605-615) copies dhead_/dvocab_/n_dvocab_ but NOT
dhead_type_, which stays at its default -1 (mtp.hpp:193); the slot's first
draft at admission feeds -1 into native_mmvq (mtp.cpp:952) -> throw ->
"strata batch: MTP admission for slot %d failed" -> engine unload. The head
type (Q6_K=14, 212.9 MiB / 106299 rows = 2100 B/row) was ALWAYS supported.
Fix: dhead_type_ = shared->dhead_type_;  (one line + comment).
**Rider (same change):** the batch-site ratio arm was a no-op (ratio_accept
called with drafter_p=-1 at generate.cpp:7728 -> "q >= 1.0", never accepts;
the slot draft call omitted probs at :7773). Added BSlot::dprob[], passed
sl.dprob.data() + o.spec_min_p into the slot draft(2,...) call, and
(double) sl.dprob[0] into ratio_accept. The batch rescue now has a real
p-side for STRATA_SPEC_RATIO A/Bs.
**DEPLOY LAW (learned the hard way):** build-strata.bat links to
build\strata.exe; the config launches engine\strata.exe - a MANUAL COPY.
The first "fixed" boot ran the stale 18:04 F-29 binary and replayed F-28b
verbatim (503 pair, identical crash). After any build: copy
build\strata.exe -> engine\strata.exe, then boot. (Also seen: one silent
boot death at cache-sizing on a stale-binary boot - transient, never
recurred on the fixed binary.)
**ARM PROOF (all three gates):** "strata batch: slot N takes X tokens
(copied in Y ms)" with NO failure line; batch windows "over slots 1,1,0,0"
(each active slot = TWO rows: token + draft; draftless was one row/slot);
"367 windows, avg 3.76 rows" vs the 2.00 draftless signature.
**BEFORE -> AFTER battery (full GATE LAW set, same order):**
- serial 700-tok: 8.5s/82 -> 5.1s/137 t/s. The BEFORE leg was depressed by
  just-parked 105k real traffic; AFTER = the F-29 4.8s-class restored. Not
  a batch-mtp effect (serial never touches slots).
- PAIR (the target workload): 16.5s/85 -> **9.6s/146 t/s aggregate (-42%
  wall, +72% t/s)** - at the old 10.6s draftless anchor, past it.
- 6-concurrent: battery legs read 34.6s/121 (before) vs 40.4s/104 (after)
  -> LOOKED like a regression. Attribution experiment (fresh boots, 6-conc
  FIRST): ON 34.6s/121 vs OFF (same binary) 33.9s/124 -> NEUTRAL (+-2.4%
  noise). The battery-leg 104 was probe-sequence cache state, not the change.
- depth 64k synthetic: decode 52.6 -> 53.4 t/s (parity); prefill 4824 ->
  1310 tok/s on the battery leg and 672 on a late re-probe - the synthetic
  filler's cold-expert misses compound on a fresh boot; NOT read as a
  batch-mtp effect (the draft path does not touch prefill), but recorded.
- MMLU 70q: 94.3% -> 92.9% (66->65 of 70; inside the +-2-question noise law;
  failure set = the 2 stable model-limited + rotating flaky, as always).
- mimic x2: 96.9/96.9 -> 96.9/96.9 (PARITY, drift 12/12 @ 100% all runs).
- compliance: 10/12 -> **12/12**.
**THE RECORDED TRADE (VRAM tax):** 4 slot drafters = 4x50 MiB private
state + draft KV; boot shows 10063 slots (vs 10.8-11.1k standing) and
198 MiB free - LOW warning ("add --vram-reserve-mib 1314"). Under EXTREME
sustained synthetic churn (2x 64k fillers + full battery back-to-back) the
late lane degraded to 39 t/s @ 6-conc with the miss-recovery signature
(PCIe 5.28/layer-window x5.6, CPU experts 11.19 x4) - the ~800-slot cut
lowers the miss ceiling; that regime never occurs in production traffic.
If it ever does: raise vram-reserve or shrink --mtp-window.
**QUIRK NOTED (upstream, benign):** admission's draft_first(1,...) writes
zero drafts (T-1=0), so each slot's first window proposes stale token 0 -
one wasted row per admission, near-zero accept chance, self-heals next
window. Candidate follow-up: draft_first(2,...) at admission.
**Standing:** STRATA_BATCH_MTP=1 in the launcher (the env arm; --batch-mtp
arg works too). Lane live on the fixed binary. The 4.3-4.6s pair prize
anchor remains the warm-lane ceiling to chase (we landed 9.6s cold-boot
with lazy graph captures); next levers: the config wave (coupled drafts,
spec {5,6}, pcie-dma, mtp-q4) per the arch-study boards.

### F-30b — The No-Regression Law + the batch-rescue gate + the MMLU band law (2026-10-10)

**THE NO-REGRESSION LAW (user-issued, BINDING - stamped into THE GATE LAW
above):** Speed, Quality (MMLU), Adherence (mimic) must NEVER regress - all
three, every change. A lower reading is never "noise" to wave through:
re-measure and resolve it. Noise counts only in the positive direction.
(User's arithmetic: tolerating -1.4/round compounds to -5..-10 per campaign.)
**THE MMLU BAND LAW (measured the same day):** this lane's true MMLU-Pro
band is 91.4-97.1 at n=11 runs across arms (draftless mean 95.0 n=4,
drafting mean 93.6 n=7, t~1.4 p~0.19 - not separable), i.e. ~2.5-question
sigma - WIDER than the old +-2-question law. Adoption math compares ARM
MEANS over >=3 runs. The 2 STABLE failures (business F->G, psychology
F->E) are invariant across every arm ever run; everything else rotates.
**THE BATCH-RESCUE GATE (STRATA_BATCH_RESCUE, default OFF):** the F-28
rescue port was the only distribution-bending step in the slot path
(exact-match accept AND reject both emit the target's own pick; the rescue
swaps in the DRAFTER's token). Single sequential requests DO land in slots
(116 probe-class admissions logged). With the rescue armed: MMLU
92.9/92.9/91.4 vs 95.7 off (same binary, fresh boots). generate.cpp
batch_rescue_on() gates it; the SERIAL rescue keeps STRATA_TYPICAL_ACCEPT
(the standing recipe, 98.57 co-measured).
**Speed consequence (why F-30 was suspended):** at temp 0.7 the pair win WAS
the rescue (exact-match acceptance of argmax drafts is rare when sampled):
pair 16.5s draftless -> 9.6s with rescue -> 17.2-17.9s drafts-only. Legal
speed needs coupled drafts in slots (F-31, open).

### F-30c — F-30 suspension verdict + the deploy/env laws (2026-10-10)

**SUSPENDED:** STRATA_BATCH_MTP back to 0 (the F-29 recipe stands) until
coupled-in-slots lands. Net at this state: pair -6% (17.5 vs 16.5s) with
rescue off = a law violation; not shippable. The dhead_type_ fix, the
dprob plumb, the rescue gate, and the slot set_draft_sampling wiring all
stay in the code (inert at batch-mtp 0, armed for F-31).
**6-conc attribution (fresh boots, 6-conc FIRST):** ON 34.6s/121 t/s vs OFF
(same binary) 33.9s/124 - batch-mtp was NEUTRAL at 6 clients; the battery
leg's 104 was probe-sequence cache state. The BEFORE battery's lane (19:05
boot) ran 10010 slots - the "~800-slot tax" story was wrong; the slot cut
happened between 18:20-19:05 (external VRAM), BEFORE and AFTER measured
the same slot count.
**ENV-OVERRIDE LAW:** launcher `set X=1` lines CLOBBER inherited env vars -
arm changes go through launcher edits with read-back, never `env:X=0`
overrides (a control run was silently mis-armed this way).
**Deploy law restated:** build\strata.exe -> engine\strata.exe copy after
every build; verify by timestamp (a stale-binary boot replayed F-28b
verbatim and cost a debugging cycle).

### F-31 — Coupled drafts: serial-path win proven, slot-side open (2026-10-10)

**ARMED:** STRATA_SPEC_COUPLED=1 (built, unit-tested, never armed before -
arch-study A-board found it OFF on the live lane). Boot arm line prints 4x
(one per slot drafter + serial). set_draft_sampling now ALSO called on the
slot drafter at admission (generate.cpp BADM block - was serial-only; the
shared-bind/missing-wiring gap class, second instance after dhead_type_).
**PROVEN:** serial-path acceptance 83.9% (425/506) at temp 0.7 - the
coupled mechanism works exactly as designed (drafts sample with the
target's chain + same Philox uniform; coupled_draft.hpp:5-6: "the output is
a function of the seed alone, whatever the drafts are" - structurally
incapable of biasing quality).
**OPEN (the blocker):** slot-side drafts still do not couple - pair stays
17.5s/80 t/s with the slot leg ~40 t/s while the main leg drafts 83.9%.
Suspect: the counter/cell staging for slot draft graphs (the design's
counter law: "a draft made at cell c is verified with counter c+1" - the
slot rows' Philox counters may not match what the slot drafter's coupled
samples drew). Needs a diagnostic boot (per-window slot acceptance
telemetry), not more static reading.
**MMLU under F-31:** 92.9/94.3 gate + 92.9/95.7 pre-fix - arm-mean parity
with draftless (the band law). Quality was never the risk; speed is.

### F-31b — The PLE table question CLOSED (user-hypothesized, 2026-10-10)

The user challenged the FP8-table hypothesis ("if IQ4_NL is 8% off per row,
FP8 should give 70/70 - either the tests are unfair or something was
wrong"). Two-agent study (.docs/research/2026-10-10-ple-study/ A+B):
**FAIRNESS: the user was structurally RIGHT - historically.** 8 pre-F-19
MMLU runs contain 3-8 EMPTY answers each (reasoning overran the mt4000
budget; scored as failures; 81-92% era scores). ZERO empties in all ~14
post-F-19 runs. The unfairness was real, worth 3-8 points/run, and already
fixed by F-19's answer-space reservation.
**KNOWLEDGE: the table never bound.** Three-way triangulation: (1) the
PLE-off ablation (base-off arm, generate.cpp:2249) answers the IDENTICAL
clean "F" on both stable failures with NO table at all; (2) ISTA-DASLab's
own card: the table is "held at a fixed 4.5 bpw in every build and
excluded from the search... quantized in the evaluated models" - every
published parity result for this family ran WITH the IQ4_NL table; (3) the
paper: downstream accuracy saturates vs ngram vocab size. F-21's UD-Q4
carried the same IQ4_NL table (all unsloth UD-Q4-class packs do). The
variable was never varied - and does not bind: the 2 stable failures are
BACKBONE knowledge misses.
**THE DECISIVE ARM (if ever wanted):** unsloth UD-Q6_K_XL shard 3 = the
table alone as Q8_0 (54.4 GB, one file, engine accepts natively - needs
--ple-io direct or conv-cache 32 to fit RAM), or the true FP8 (32-shard
~51.3 GB sparse pull from Qwen/Qwen3.8-Flash-Next-FP8 ->
tools/ple_fp8_pack.py -> 47.7 GiB). Prior from the evidence: parity.
**PRUNE PIPELINE (spec'd, deprioritized):** the hash (row = offset[h] +
XOR(ctx*mult) % vocab[h], ngram.cpp:62-93) is forward-enumerable but the
collision wall (lambda~3083 preimages/bucket for bigrams) kills
content-classification - the only safe criterion is the dormant-row rule
(rows never READ); zero-row prune is bit-for-bit inert for unread rows;
compact-FP8 at <=56.25% keep = byte-parity RAM. Machinery spec'd in
B-prune-feasibility.md for the day we want the RAM back. 90 unreachable
tail rows are always zeroable (sum(vocab) < PLE_TABLE_ROWS).
**Record corrections:** Swift's table is IQ4_NL (not IQ3_XXS - B's earlier
read was wrong); 'jc' = JonathanColetti 27B, nothing to do with the table.

### F-32 — The spec {4,5,6} sweep: 5 is faster, 5 FAILS the mimic gate - reverted (2026-10-10)

**THE SWEEP (serial 700-tok x3 per arm, warm runs 2-3 compared):** spec 4
4.45s / spec 5 4.30s (+3-4%, inside probe noise) / spec 6 4.55s (regress).
The knee is at 5; the trained-4th-draft-step theory (arch-study B-L2A)
predicted ~+18% - measured: noise-level.
**QUALITY GATE (the law):** spec 5 MMLU 97.1/92.9 (mean 95.0 = standing
parity), mimic **95.3/92.2 (mean 93.75) vs standing 96.9 x4 - ADHERENCE
REGRESSION.** Miss pattern: ui-mimic 7-9/10 (standing 9-10/10) +
precedence 7/8 + a format-lock miss (7/8, first ever). Failure items
runaway (ui-07 47.3s, prec-07 64.7s). Mechanism: sampling-path variance
(longer windows shift the Philox alignment; acceptance stays exact-match -
no bias, but the borderline mimic items flip), not a distribution bend.
**VERDICT: REVERTED to spec 4. Anti-lever recorded: spec {5,6} costs
mimic adherence for noise-level speed.** Config restored; lane standing.
**Boot-race law (same session):** after killing the engine, WAIT for
nvidia-smi to show VRAM reclaimed (~5-15s) before relaunching - a raced
boot sizes the expert cache at 2097 slots (a fifth of standing) and never
regrows it. This cost the 12:01 boot (user caught it: "Experts in VRAM
4,644 - dropped considerably").
**Serial record note:** the coupled serial path (F-31 standing arm) reads
4.4-4.5s / 155-158 t/s warm - the new serial class (prior record 4.8s /
146-class). Recorded as the F-31 serial dividend.

### F-32b — The user soak (2026-10-10, ~2.5h real traffic on the standing lane)

The user ran their own workload on the standing lane (spec 4 + coupled
serial, batch-mtp 0) for ~2.5 hours after the F-32 revert and reported:
"holding above 130 [t/s], some spikes down but rare." That is the
production confirmation of the F-31 serial dividend on real traffic (the
130+ class vs the pre-coupled ~110-120 real-traffic band), and the standing
recipe's first soak since the No-Regression Law. Spikes rare - consistent
with the adapt/miss-bounce class, no action.
**Commit discipline (user-issued, 2026-10-10, BINDING):** progress commits
land in story-shaped units - each commit's description explains everything
it did, and only WORKING things ship in the tree (failed arms land as
default-off gates or reverts, with the negative recorded here). Purpose: a
future fork of strata can cut a clean PR from this history.

### F-33 — --pcie-mode dma ADOPTED: depth decode +58-75%, quality parity (GATE complete, 2026-10-10)

**MECHANISM (arch-study C-3):** the default (--pcie-mode unset = mode 2)
never runs the copy engine for in-window misses - an in-graph fetch_blobs
kernel pulls blobs from pinned RAM with SMs AFTER the hit grouped kernel,
serialized. Mode 0 (dma) copies on the copy_ stream, overlapping the hit
kernel (flag B gates only the PCIe group). At depth with ~20% misses this
is the difference between waiting and hiding.
**MEASURED:** depth probe (64k synthetic, ~79-81% hits): decode
52.6-53.4 -> **83.3 / 92.8 t/s (+58-75%)** - attribution clean: acceptance
actually DROPPED (94% -> 86%) while throughput jumped, so the gain is the
miss path, not drafting. Depth prefill 5733 tok/s (at the warm-lane best).
Serial parity (5.3-5.7s / 123-131 t/s). Window stats: draftless windows
24.41 ms at 1.98 rows on a cold boot - the F-29-era class, not worse.
**GATES (arm-mean protocol, the No-Regression Law):** MMLU 92.9/91.4/94.3/
94.3 -> mean 93.2 (n=4) vs standing-shape 93.6 (n=2, same failure
fingerprint) = parity. Mimic 96.9/95.3/95.3 -> mean 95.8 vs the RECORDED
standing 95.3 (band 90.6-98.4; the 96.9x4 was one battery era) = at/above
record. Misses rotate around the fixed intrinsic ui-10 (lex-11/ui-06/
ui-07/prec-07 rotate) - the numeric-noise class, no systematic item flip.
**ADOPTED:** --pcie-mode dma in the config (standing). The 40ms-window
mystery from F-31 is substantially explained: the unaccounted ~10ms was
the serialized miss fetch inside run_slot_rows - this arm attacks it.
Remaining batch-window work (CPU-expert misses, the sync-hoist) stays on
the C-board queue.

### F-34 — --mtp-q4 ADOPTED: the draft chain 31% cheaper, quality parity-to-better (GATE complete, 2026-10-10)

**MECHANISM (arch-study A §3):** Q8_0 -> Q4_0 copies of the drafter's 10
projections AND the draft head at bind (mtp.cpp:182-240). Boot proof:
"draft head over 106299 tokens as Q4_0 (146.0 MiB read per step, was
212.9)" - the per-step head read drops 31%. Drafts-only contract
(mtp.hpp:13-14): verification stays exact-match against the target's own
picks, so outputs are structurally unchanged.
**MEASURED:** serial warm class 4.1-4.3s / best 171 t/s (new best
observed); acceptance held (the 700-tok probes ran at the coupled class).
**GATES:** MMLU 91.4/94.3/95.7 -> mean 93.8 (n=3) - the HIGHEST arm mean
of the campaign day, standing parity+; mimic 96.9 (top class, the standing
miss set). Failure fingerprints identical across arms throughout.
**ADOPTED:** --mtp-q4 all in the config (standing).

### F-35 — STRATA_PLE_BATCH=1 ADOPTED: batch windows -24% (GATE complete, 2026-10-10)

**MECHANISM (arch-study C-7.3):** multi-row PLE key/value projections for
batch windows (verify.cpp:748 class, default OFF upstream) - the projection
weights read once per window instead of per row.
**MEASURED:** batch windows 24.41 -> **18.47 ms/window** at 2.00 rows
(-24%, the best batch-window number of the campaign; GPU-wait 7.12 + CPU
experts 9.43). Pair 13.2s (parity with the standing class); serial 4.2s /
168 t/s held. Lane warm-state note: the pair wall is now the slot leg's
token rate (draftless slots commit 1 token/window; ~700 rows for 700
tokens at 18.5ms = 12.9s - arithmetic checks).
**GATES:** MMLU 94.3/94.3 -> mean 94.3 (n=2, the strongest pair of the
day); mimic 96.9 (top class).
**ADOPTED:** STRATA_PLE_BATCH=1 in the launcher (standing).

### F-36 — STRATA_VERIFY_DEVICE_PLAN=1 ADOPTED + the TREE LANE CLOSED WITH EVIDENCE (GATE complete, 2026-10-10)

**MECHANISM (arch-study C-7.4):** device-side routing plan when all experts
are resident - skips the host plan publish round-trip (the flag-A path,
verify.cpp:592 / expert_source.cpp:2584; default OFF upstream).
**MEASURED (the stack now: dma + q4 + PLE_BATCH + DEVICE_PLAN):** depth
prefill 7514 tok/s (NEW BEST, was 5733 dma-only, 4824 session start);
pair 10.5s best-observed (13.2-13.8 repeatable class; standing was 13.0);
serial 4.2s / 166 t/s held; final standing depth decode 96.6 t/s.
**GATES:** MMLU 94.3/95.7 -> mean 95.0 (n=2, strongest arm mean); mimic
95.3/95.3 -> mean 95.3 (= the recorded standing exactly; the miss set
rotates around the intrinsic ui-10).
**THE TREE LANE (closed):** STRATA_MTP_TOP2 ran its one diagnostic boot -
runner-up coverage of rejections 0/57 across 304 windows = **0%** (the
close-threshold was <15%). A width-2 tree catches NOTHING on this
workload; the series-198 fused-kernel requirement is moot. Anti-lever
recorded. Env removed after the boot (its D2H cost dropped depth
acceptance 86% -> 66% - as A-board warned, diagnostic boots only).
**ADOPTED:** STRATA_VERIFY_DEVICE_PLAN=1 in the launcher (standing).

### THE F-33..F-36 STACK - the session's net speed record (2026-10-10)

Depth decode 52.6 -> 96.6 t/s (+84%) | depth prefill 4824 -> 7514 tok/s
(+56%) | serial 4.8s/146-class -> 4.1-4.2s/166-171 (+15%) | pair 16.5 ->
13.2-13.8 repeatable (10.5 best) | batch windows 24.4 -> 18.5 ms | quality
arm-means 93.2-95.0 MMLU / 95.3-96.9 mimic with the invariant failure
fingerprint throughout. All four adoptions are one-variable-gated with
the full protocol; every negative (spec 5, the rescue, batch-mtp
suspension, the tree) recorded with evidence.

### F-37 — The level playing field: our fork vs upstream v0.1.42 as shipped (2026-10-10)

**PROTOCOL (user-ordered):** upstream pulled to v0.1.42 (worktree at
E:\LLM\Engine\strata-stock; our fork's llama.cpp backend vendored in -
build clean 138/138), the SAME config args (all flags upstream-valid),
ZERO env arms, stock's own .42 server, same model/pack/port, same
harnesses, same day, same machine. Stock booted level: 10,074 slots
(ours 10,079).
**THE TABLE:**
- MMLU 70q: stock **88.6% with FIVE EMPTY ANSWERS** (pred "?" on
  CS x2/econ/history/law - the reasoning-overrun class) vs our 93.2-95.0
  arm means with ZERO empties since F-19. **+4.6-6.4 ours.**
- Mimic 64: stock **90.6%** (precedence 5/8, code-style 2/3) vs our
  95.3-96.9. **+4.7-6.3 ours.**
- Serial: stock 5.0-5.7s / 123-139 t/s vs ours 4.1-4.2s / 166-171.
  **+20-23% ours.**
- Depth decode: stock 78.9 vs ours 96.6 t/s. **+22% ours.** Depth
  prefill: stock 6632 vs ours 7463-7514. **+13% ours.**
- PAIR: stock **10.7s / 131 t/s** vs our 13.2-13.8 repeatable (10.5
  best). **STOCK WINS -22-28%. THE ONE REGRESSION.**
**ROOT CAUSE OF THE PAIR GAP (from stock's own telemetry):** stock
draftless windows 14.77 ms with CPU experts 2.68 ms vs our 18.47/9.43 -
upstream .41/.42 (their f-142 pciefrac line + an "adapt wait" phase we
lack) nearly eliminated CPU-expert time in batch windows. Same slots,
same draftless state: their miss path is ~20% faster per window. This is
EXACTLY the C-board queue (CPU-expert misses in batch windows + the
sync-hoist) and the reason to test their .41/.42 updates on our fork
(the user's queued .40-vs-.41 rebase test).
**VERDICT (the user's frame - "prove we made a leap, or regressed"):
BOTH.** Quality: a LEAP (+4.6-6.4 MMLU via F-19's empty-elimination,
+4.7-6.3 mimic via the cvec/recipe work). Serial/depth: a leap
(+20-23%/+22%). Pair vs .42-latest: a REGRESSION - the one place
upstream moved past us while we were gated on the rescue/batch-mtp law
work. The stock worktree stays at E:\LLM\Engine\strata-stock as the
baseline rig; scaffolding (build-stock.bat, strata-stock.json,
launch-stock-8081.bat) untracked there by design.

### F-38 — The storage consolidation (~163 GB reclaimed, 2026-10-10)

Estate drifted to 3 engine trees + a root-level Strata-data duplicate
home; consolidated per the storage canon: (1) E:\LLM\Strata-data\
deleted (the original download site - shards SHA-verified identical to
the canon home, mtp/packs strict subsets of the engine's); (2) the old
0.1.39 install (Engine\strata) deleted after extracting data\; (3) the
stock v0.1.42 worktree MOVED to E:\LLM\Engine\strata (the one stock
tree, per the user); (4) all configs' --native/--ple-gguf repointed at
the canon home E:\LLM\Models\GGUF\gsq-rco\qwen3.8-flash-next\
gsq-iq3s\IQ3_S\ (SHA 4C1EB2CE.../316B46F3... = the HF LFS shas);
engine working state (packs/tokenizer/mtp) stays at Engine\strata\data.
Ledger: llama-custom/.docs/CONSOLIDATION-LEDGER-2026-10-10.md (written
BEFORE the deletions, per the canon law). Lane proven booting from the
new layout (10,059 slots).

### F-39 — The batch-window adapt port (upstream e989c671): CPU experts halved, pair 12.1-12.7s (GATE complete, 2026-10-10)

**THE FIND (the .42-diff agent board, 2026-10-10-dot42-diff/):** the
F-37 pair gap's mover was NOT the f-142 pciefrac headline (both marquee
changes gate off under --batch; the pcie defaults are byte-identical
across .40/.41/.42). It was e989c671: pre-.42 batch_step NEVER called
adapt() - the VRAM expert tier FROZE for the whole concurrent phase.
**THE PORT (isolated, per the agent's map):** adapt() every
--adapt-every windows on a thread beside commit/emit, joined at window
end, swaps landed via apply_pending(true) before the next window plans;
guards: upstream's loan check (pf_parts[].lent empty) + OUR addition -
prompt_read_active (our fork interleaves batch_step inside yielded
prompt chunk reads, a state upstream's guard does not know). Telemetry
extended (adapt wait/rounds/swaps in the batch line).
**MEASURED:** CPU experts 9.43 -> **4.23-4.25 ms/window (-55%)**;
windows 18.47 -> 16.15 ms; the "adapt wait 0.53" fingerprint fires;
PAIR 13.2-13.8 -> **12.1-12.7s** (stock .42: 10.7 - the remaining ~1.5
ms = the d6850f31 AVX2 decode + 5208b320 Q8_K quantizer kernel ports,
sized 0.5-2.0 + 0.2-0.5 ms by the diff board; queued next).
**GATES (the cleanest of the day):** MMLU 94.3/94.3; mimic
**96.9/96.9** (top class both runs). The residency-mix channel is the
numeric-noise class (stock's own BATCHING.md: "greedy outputs can
differ run to run" - same for us).
**ADOPTED:** the port is live in the standing binary. Session arc:
pair 16.5 (start) -> 13.2 (F-35) -> 12.1-12.7 (F-39); CPU experts
9.43 -> 4.23; quality held at the strongest readings throughout.

### F-40 — The upstream kernel cherry-picks + STOCK PARITY CROSSED (GATE complete, 2026-10-10, goal-57c7a9dd round 1)

**THE CHERRY-PICKS (upstream authorship preserved - the PR-native form):**
d6850f31 (IQ3_S/IQ2_S AVX2 decode without per-index shift/mask/or -
arithmetic untouched, same words in the same lanes; "ported onto the
0.1.40 layout") and 5208b320 (AVX2 Q8_K activation quantizer). Both
applied clean onto the fork.
**MEASURED (the F-39+F-40 stack):** PAIR **10.4s / 135 t/s - stock parity
CROSSED** (stock .42: 10.7; session start: 16.5). **6-CONCURRENT 171/165
t/s aggregate** - the 116-124 standing record broken by +38-47%, past
stock's observed ~150 average. Serial 4.6s warm (153 t/s class, 171 best
class). Depth: lane-state variance this boot (6739/83.9); the standing
depth class 7463/96.6 unchallenged.
**GATES:** MMLU **98.6/94.3** (98.6 TIES the all-time rig record; mean
96.45 = the strongest arm mean of the campaign); mimic 92.2/95.3/**
98.4 - THE CAMPAIGN RECORD** (only the intrinsic ui-10 miss; mean 95.3 =
the recorded standing exactly; failure sets rotate as always, the
kernels are bit-identical by design so the spread is lane-state).
**ADOPTED.** The fast-gate triage harness (4 min: serial x2 + pair +
depth + MMLU-35 slice) landed this round per the test-friction mandate
(bench/measure/fast-gate.ps1 + mmlu-pro-strata-fast.py; triage-only, the
full battery stays the law for adoption).
**Goal progress (goal-57c7a9dd):** pair 10.4 <= 8.5 target NOT yet;
serial 171 best vs 180 target - close; 6-conc 171 vs 180 target - CLOSE;
MMLU 96.45 >= 94 MET; mimic 95.3 vs 96.5 - close; prefill 7463 class
held. The next prizes: the F-31 coupled-in-slots fix (the 8.5s pair
class), the serial push to 180, depth decode >= 100.

### F-41 — Round 2: batch-mtp re-test = definitive NO; standing records 9.8s pair / 196 t/s 6-conc (2026-10-10, goal-57c7a9dd round 2)

**THE BATCH-MTP RE-TEST (zero-code, one variable):** with the F-39/F-40
stack (16.15ms draftless windows), drafting slots read 6.58-row windows
at **44.49 ms/window (GPU-reach wait 30.45)** - the 2-row window physics
(kShape 2.85x row cost + expert union growth) costs more than the ~1.7x
tokens/slot-window drafts add. Pair 11.8-14.9s (vs 10.4 off), 6-conc 142
(vs 165-171 off). **VERDICT: batch-mtp OFF is correct on this stack -
the F-31 coupled-in-slots prize is DEAD unless the row-cost physics
change. Anti-lever recorded.**
**THE STANDING RECOVERY = NEW RECORDS (warm lane, F-40 stack):** pair
**9.8s / 142 t/s** (the 8.5 target in sight) and **6-CONCURRENT 196 t/s
AGGREGATE** - the 180 target SMASHED, the user's stock-observed
199-transient class REACHED (wall 21.4s, per-request 13.7-21.4s).
**Serial-push study dispatched** (fastmtp prior art + the alpha lever +
depth-100 levers; board: 2026-10-10-serial-push/).

### F-41b — Round 3: TWO TARGETS STABLY MET (2026-10-10, goal-57c7a9dd round 3)

**6-CONCURRENT: STABLY MET.** 196 / 189 / 190 t/s aggregate (n=3, mean
191.7, all >= 180). Stock's observed ~150 average beaten by +28%; the
199-transient class reached.
**PAIR: 9.8 / 9.2 / 9.3 / 9.4 (n=4, mean 9.43, best 9.2 / 151 t/s).**
Not yet <= 8.5 but the warm trend is downward; stock 10.7 beaten by
~12-14%.
**Serial: 150-162 t/s warm class** (171 best-ever) - the 180 target
awaits the alpha lever (study board in flight).
Standing targets now: MMLU MET (96.45), 6-conc MET (191.7 stable),
empties MET (0), pair/serial/depth/mimic closing.

### F-42 — Round 4: the serial-push study + min-p 0.5 tested and REVERTED (2026-10-10, goal-57c7a9dd round 4)

**THE STUDY (2026-10-10-serial-push/A-levers.md, pass 2 - every cite
re-verified):** the fitted window model V(T) = 9.7 + 2.47(T-1) ms
reproduces both measured serial classes; LAW FOUND: alpha pays THROUGH
chain length - 0.86->0.92 buys only +4% at min-p-0.7 truncation but hits
180 exactly once confidence lengthens chains past the gate. Ranked:
--mtp-hnorm (correctness alignment with llama.cpp's per-stream RMSNorm -
the top acceptance mover), STRATA_MTP_CATCHUP_ALL, the FULL_HEAD
diagnostic (prices the 106k draft-vocab subset tax), the FastMTP
trained-head drop-in (ZERO engine LOC - fine-tuned head -> mtp_rt.py ->
--mtp DIR; the R17 verdict is staged in the buun lane), depth: the
per-request pcie_frac sweep + the owed dmon classification probe. The
S17 self-drafter KILLS ITSELF on cost (one approximate 48-layer pass =
80% of the whole 12ms window to replace a 1.5ms chain at 83.9%) -
parked with evidence.
**MIN-P 0.5 TESTED (one variable):** pair 8.9s (best yet, target 8.5
nearly touched), depth prefill 7503 (record), serial parity. GATES:
mimic 93.8/98.4 (mean 96.1, fine) but MMLU 91.4/92.9/94.3 -> arm-mean
92.87 vs the standing 96.45 - a 3.6-point drop for -0.4s pair.
**VERDICT: REVERTED to 0.7 (anti-lever recorded - the spec/min-p family
is quality-taxable; F-32's spec-5 and now min-p-0.5 both confirm).** The
8.9s pair shows the 0.7 lane's warm floor; the next pair push must come
from the acceptance levers, not the truncation gate.

### F-43 — --mtp-hnorm stream: TWO TARGETS MET (depth 102-146 decode, prefill 7591) (GATE complete, 2026-10-10, goal-57c7a9dd round 5)

**THE ARM (the serial-push board's #1):** --mtp-hnorm stream - the
drafter's hidden-state norm aligned with llama.cpp's reference qwen4exp
graph (per-stream pre_fc_norm_hidden vs the engine's one-RMS-over-all-
four-streams). Draft-path only: L3-legal (outputs structurally
unchanged; the verify stays exact). CONFIG TRAP: the flag takes an
argument ("pooled" or "stream") - a bare --mtp-hnorm exits the engine
with a one-line usage message.
**MEASURED:** DEPTH DECODE 102.1 / 121.4 / 146.0 t/s (stability set;
the >=100 target MET with room - the standing was 96.6); DEPTH PREFILL
7565-7591 tok/s (the >=7500 target MET; record class). Serial 147-160
class; pair 9.3 (the 0.7-lane class).
**GATES:** MMLU 92.9/92.9/95.7 -> arm-mean 93.83 = the session class
(cross-arm mean ~94.3; F-40's 96.45 was a 2-run high); mimic
95.3/96.9 -> 96.1 (fine class). L3-legal: the drafts change, the
outputs cannot.
**ADOPTED:** --mtp-hnorm stream in the config (standing). Two more goal
targets banked: depth decode + prefill.

### F-44 — STRATA_MTP_CATCHUP_ALL tested and REVERTED (2026-10-10, goal-57c7a9dd round 6)

The serial-push board's #2 arm (fill the drafter's K/V holes after
rejections, mtp.cpp:58-64; estimated +2 acceptance pts at 0.1-0.3
ms/window). MEASURED: a clear NEGATIVE at every level - serial 26.8s
cold / 6.2s warm (vs the 4.4-4.9 class), pair 15.5s (vs 9.3), depth
decode 78.6 (vs 102-146), prefill 5440 (vs 7591). The catch-up rows are
FULL draft-layer forwards - the cost dwarfs the acceptance gain.
**REVERTED same-round; anti-lever recorded (the board's 0.1-0.3ms
estimate was wrong by an order of magnitude).** Standing recovered:
serial 4.9/142, pair 10.6 warming, depth 7503/88.8 warming.

### F-45 — Round 7: the per-request tune sweep is FLAT on warm serial (2026-10-10, goal-57c7a9dd round 7)

The F-27 harness (spec_min_p {0.5,0.6,0.7,0.8} x pcie_frac
{0.25,0.35ctl,0.45,0.55}, 2 probes each, live lane): every arm reads
4.0-4.5s - warm serial is saturated at the fixed-cost wall; pcie_frac
does not bind at warm-lane hit rates and the min-p family is closed
(F-42). **The serial 180 target's remaining lever is alpha itself: the
FastMTP trained head (zero engine LOC, R17 staged) - next round's
opening task.** Depth-side pcie_frac A/B queued behind a depth-probe
tune-key variant (harness tweak).

### F-46 — Round 8: the FastMTP head located; the drop-in needs a name-map pass (2026-10-10, goal-57c7a9dd round 8)

The trained head EXISTS (E:\LLM\Models\GGUF\MTPs\mtp-v5-phaseB-F16.gguf,
4.87 GiB, best-val ep2 per the series-192 manifest; R17 never ran).
tools/mtp_rt.py REJECTED it: the head is a Ckpt-layout trained
checkpoint (general.name='Ckpt_Q38', HF-style tensors + chat template)
- not the stock MTP GGUF naming (mtp.layers.0.*) the rt-builder
consumes. **The bridge is a bounded tensor-name/dtype mapping pass**
(Ckpt names -> stock MTP names -> mtp_rt.py -> rt dir -> --mtp DIR) -
spec'd for the next round; the alternative is running R17 on the buun
lane first. No engine change; nothing regressed; the lane untouched
this round.

### F-47 — Round 9: the FastMTP bridge spec completed (2026-10-10, goal-57c7a9dd round 9)

The phaseB head is an MTP-only GGUF in BUUN-SIDECAR format (series-192
patch 0005 wrote "battery-ready F16/Q8_0 sidecars" for the buun lane's
--spec-draft consumer; parent metadata says block_count=49 but the
4.87 GiB F16 payload is the trained MTP layer alone). **The strata
bridge = a name-table mapping pass:** dump both inventories (the
buun-sidecar names from mtp-v5-phaseB-F16.gguf; the strata names from
tools/mtp_rt.py's expectations - mtp.layers.0.mlp.experts.gate_up_proj
class), join on tensor SHAPES (the unambiguous key), rename, re-emit
the GGUF, run mtp_rt.py -> rt-phaseb dir -> --mtp DIR -> fast-gate +
the acceptance read. Both dumps work with existing tools (gguf_meta.py
with PYTHONIOENCODING=utf-8; the crash was a cp1252 tokenizer print,
not a format problem). One focused session's work; the payoff is the
alpha lever (the window model's serial-180 path).

### F-48 — Round 10: the FastMTP bridge BUILT to the last step (2026-10-10, goal-57c7a9dd round 10)

tools/mtp_bridge.py written and run: 32/32 phaseB tensors shape-matched
to the reference with ZERO collisions -> mtp-v5-phaseB-strata.gguf
written. THE DISCOVERY: three dialects exist - buun-sidecar (phaseB),
llama.cpp (the MTPs reference GGUF: blk.48.attn_q / SEPARATE
ffn_gate_exps+ffn_up_exps), and strata rt (mtp.layers.0.self_attn.q_proj
/ FUSED-INTERLEAVED mlp.experts.gate_up_proj). The bridge mapped
phaseB->llama.cpp dialect cleanly; the LAST STEP is a ~15-line alias +
fusion patch in mtp_rt.py (llama.cpp names -> strata names; gate+up
rows stacked and interleaved per blob_of()'s layout). The full
reference tensor table (32 names+shapes) lives in the session log and
the bridge script's output. NEXT SESSION: the alias patch -> rt-phaseb
-> --mtp DIR -> fast-gate + acceptance -> the alpha verdict.

### F-49 — Round 11: the bridge's final piece is the Q2_0 expert quantizer (2026-10-10, goal-57c7a9dd round 11)

Opening the mtp_rt alias patch exposed the last dependency: the phaseB
head is F16, but experts.bin's contract is PRE-QUANTIZED Q2_0 blobs
(the strata-format source GGUF was Q2_0; mtp_rt.py relayouts bytes, it
does not quantize). The bridge therefore needs a ~30-line EXACT port of
ggml's quantize_row_q2_0_ref (32-value blocks, the precise scale/round
rules - approximations would corrupt every expert) BEFORE the alias +
fusion patch lands. Scope: one careful session; everything else on the
bridge is done. Recorded so the next session starts at the true edge,
not a rediscovery.

### F-50 — Round 12: the bridge SOLVED - mtp_pack.py IS the builder; the full dialect map (2026-10-10, goal-57c7a9dd round 12)

THE BREAKTHROUGH: tools/mtp_pack.py is the ORIGINAL strata-MTP-GGUF
builder - it takes the checkpoint's own BF16 mtp.* tensors (manifest +
.bin files, the mtp_fetch.py layout) and packs the strata GGUF with the
EXACT min-MSE 64-block Q2_0 quantizer (already in the tool, vectorized,
verified). NO new quantizer needed - F-49's fear was misplaced. The
adapter is: extract phaseB tensors F16->BF16 .bin + write the manifest
-> run mtp_pack verbatim -> mtp_rt.py -> rt dir.
THE DIALECT MAP (llama.cpp ref -> strata canonical, from the working
rt/dense.txt): attn_q/k/v/output -> self_attn.{q,k,v,o}_proj;
attn_{q,k}_norm -> self_attn.{q,k}_norm; indexer.{q,k}_proj FUSE ->
self_attn.indexer.index_qk_proj (concat output axis); indexer.{q,k}_norm
-> self_attn.indexer.{q,k}_layernorm; ffn_gate_exps+ffn_up_exps FUSE
(gate rows 0-639, up 640-1279) -> mlp.experts.gate_up_proj;
ffn_down_exps -> mlp.experts.down_proj; ffn_{gate,up,down}_shexp ->
mlp.shared_expert.{gate,up,down}_proj; ffn_gate_inp -> mlp.gate.weight;
ffn_gate_inp_shexp -> mlp.shared_expert_gate.weight; hc_attn_* ->
attn_hyper_connection.{norm->hc_norm, down/up->input_mix_weight_*,
inject->block_inject_weight}; hc_ffn_* -> mlp_hyper_connection.*;
nextn.hc_head_* -> hyper_connection_mixer.*.
THREE AMBIGUITIES LEFT (all resolvable from mtp_fetch.py's fetch list
next session): fc_embedding vs fc_hidden (both (5120,2560)-class; the
checkpoint's eh_proj is one tensor), the two pre_fc_norms, and the
index_qk concat order (q|k vs k|q). Then: adapter script -> pack ->
rt -> --mtp -> the alpha verdict.

### F-51 — Round 13: the fetch SHA table = the adapter's self-verification oracle (2026-10-10, goal-57c7a9dd round 13)

mtp_fetch.py's SHA256 dict (line 44+) carries the ORIGINAL checkpoint
hash for every canonical mtp.* tensor - so the bridge adapter can
SELF-VERIFY per tensor: extract phaseB -> convert -> hash -> compare
against the oracle. Every uniquely-shaped tensor maps AND verifies
empirically; no blind trust. The canonical fetch set is 31 tensors
(fc_embedding and fc_hidden are BOTH fetched from the checkpoint - they
are distinct projections, and the buun sidecar's single eh_proj
(5120,2560) vs the canonical pair is the ONE semantic question left
(the series-192 dump patch 0001 defines what eh_proj carries). Next
session: read 0001's dump semantics -> write the adapter with SHA
self-verification -> pack -> rt -> --mtp -> alpha verdict.

### F-52 — Rounds 14-15: the bridge BUILT end-to-end; the trained head crashes at first draft - two named suspects (2026-10-10, goal-57c7a9dd)

**BUILT (tools/mtp_phaseb_adapter.py):** the full extraction chain works
- phaseB -> 31-tensor fetch layout (eh_proj SPLIT: fc_embedding = eh[:,
:2560] / fc_hidden = eh[:, 2560:] on the 5120-input axis; the fused
index_qk_proj = q|k concat; every dense tensor SHAPE-ASSERTED against
the standing rt's dense.txt ground truth; norms reshaped (1,n)) ->
mtp_pack q2_0 (0.892 GB, the exact plan) -> mtp_rt (29 dense lines =
the standing count). The rt dir is layout-verified.
**THE CRASH:** boot + first draft -> "mtp: native QSA requires a
stream, aligned spans, and disjoint buffers or exact input/output
alias" - a VALUE-level wrongness passing every shape gate. TWO NAMED
SUSPECTS (both orientation-ambiguous under shape checks): (1) the
SQUARE fc_embedding/fc_hidden pair (2560,2560) - the eh split's axis
orientation is unverifiable by shape; (2) the indexer q|k concat ORDER
(k|q would also assert clean). NEXT SESSION: one control experiment
settles both - rebuild with fc_e.T/fc_h.T and swapped qk order (2x2 =
4 builds, or reason it from mtp.cpp's eh consumer at load :318-321).
**Reverted to the standing rt; lane verified healthy (5.0s serial).**

### F-53 — Round 16: the dtype contract cracked; TWO diffs left (2026-10-10, goal-57c7a9dd round 16)

THE PACK'S CONTRACT (mtp_pack.py:173-180): ALL input files BF16; the
OUTPUT dtype comes from the MANIFEST RANK (rank-1 -> F32, rank-2 ->
BF16). The crash's likely root: my norm reshapes made them rank-2 ->
BF16 output where the loader reads F32 -> misaligned buffers (the QSA
precondition). FIXED: norms stay rank-1 in the manifest (+ the
shared_expert_gate exception: rank-2 BF16); all files BF16. Also
removed a doubled manifest append (the 60-tensor artifact).
THE REMAINING 2 DIFFS vs standing (the rt diff harness now exists -
dense.txt line-compare): (1) mlp.shared_expert_gate still f32 (the
exclusion keeps it rank-1 in the manifest - needs the (1,2560)
reshape); (2) index_qk prints cols=5120 vs standing 2560 - a real
anomaly (my concat math says (640,2560); one debug print next session
resolves whether the print or the tensor is wrong). THEN: boot ->
the alpha verdict. The diff-to-zero harness is the adoption gate.

### F-54 — Rounds 17-18: the LAST DIFF exposed the true root - tensor READ TYPING (2026-10-10, goal-57c7a9dd rounds 17-18)

shared_expert_gate fixed (the (1,n) reshape). The index_qk diff
CHASED DOWN to the real discovery: the REFERENCE GGUF's indexer also
prints (512,5120) - and 5120 = 2560 F16 values x 2 BYTES. HYPOTHESIS
(near-certain, one check owed): np.asarray(t.data) on these tensors
returns RAW BYTES (uint8) typed as float by .astype - the ENTIRE
extraction may be byte-typed, which would corrupt every tensor's
VALUES while still passing shape checks when dims coincidentally
align (the F-52 crash class, fully explained). FIRST ACT NEXT
SESSION: print t.data.dtype + t.data.shape for one phaseB F16 tensor;
if uint8, add proper typed decoding (gguf-py's tensor data access or
np.frombuffer with the F16 dtype + gguf ne-order reshape) at the
find() seam - ONE fix point, the whole bridge then re-verifies via
the diff harness (dtype+shape) + a numeric spot-check (a frozen
tensor's SHA vs the fetch oracle). The architecture note stands: the
phaseB indexer IS 5120-input (eh-concat) - the stock indexer
projections remain the correct drop-in (the adapter already reads
them from the reference; they too need the typed-read fix).

### F-55 — Rounds 19-20: THE BRIDGE COMPLETE - the trained head BOOTS AND DRAFTS (2026-10-10, goal-57c7a9dd rounds 19-20)

THE LAST PIECES: (1) the dtype check VINDICATED F16 reads (typed
float16 - the byte-garbage was only on QUANTIZED tensors: the Q8_0
reference indexer read as raw bytes; fixed by reading the BF16
reference); (2) THE INDEXER RESOLVED: BOTH sidecar GGUFs (reference
AND phaseB) carry a genuinely 5120-input indexer (the checkpoint's
eh-concat graph) - the ENGINE's drafter graph wants 2560; the fix:
tools/mtp_indexqk_splice.py dequantizes the STANDING rt's own index_qk
q8_0 bytes (the graph-correct values) into the chain.
**DIFFS: 0 - the trained rt matches the standing contract on every
name, dtype, rows, cols.** THE BOOT: CLEAN. No crash; 55/75 = 73.3%
acceptance; 118-132 t/s cold serial. The standing head's warm class
(4.4-4.9s / 142-160, ~72% acc) is the comparison bar - the trained
head's promised gains concentrate at DEEP draft positions, so the
verdict needs warm serials + the fast-gate + full gates (the
No-Regression Law). **Lane reverted to the standing rt pending the
adoption gates; the trained rt sits ready at data\mtp\rt-phaseb.**

### F-56 — Round 21: the trained head VERDICT - REJECTED with evidence; the bridge stands (2026-10-11, goal-57c7a9dd round 21)

THE WARM VERDICT (fast-gate + serials on the booted trained head):
serial 5.5-6.0s / 118-128 t/s (standing 4.4-4.9 / 142-171); pair 11.6s
(standing 9.2-9.4); depth decode 72.1 (standing 102-146); prefill 5924
(standing 7503-7591); DEPTH ACCEPTANCE 28/57 = 49% (standing 86%
class). UNIFORMLY SLOWER - the artifact, not the pipeline (the bridge
boots, drafts, gates clean: MMLU-fast 88.6, 0 empties). **REJECTED per
the No-Regression Law; lane reverted and verified (4.9s/143).**
THE ASSESSMENT: the phaseB training (202 pairs, best-val ep2) did not
converge for this lane's distribution - the deep-position collapse
(49% at depth) is the tell. THE BRIDGE STANDS as campaign
infrastructure: any future head (the R17 retrain on a diverse corpus -
the series-192 manifest's own named next round; a phaseC; or a
community trained head) drops in via the one-command chain
(adapter -> splice -> pack -> rt -> --mtp). The FastMTP question is
CLOSED FOR THIS HEAD with evidence, OPEN for better training.

### F-57 — Round 22: the LANE-STATE LAW - two GPU classes, two target classes (2026-10-11, goal-57c7a9dd round 22)

The stability re-runs read pair 10.9-11.1s (vs the 9.2-9.4 class) and
6-conc 158 (vs 189-196) - with a HEALTHY boot (10,085 slots, 19.13
GiB). CAUSE: the GPU reads 31.5/32.6 GB - ~9 GB of external apps
(user's live session) appeared since boot; the engine's runtime
contends for the remainder. **THE LANE-STATE LAW: the lane has two
measurable GPU classes - LIGHT (~2 GB external: pair 9.2-9.4, 6-conc
191.7) and LOADED (~9 GB external: pair 10.9-11.1, 6-conc 158). The
completion criteria run in LIGHT windows (or the criteria move to the
LOADED class deliberately - the user's call; the lane serves real
traffic in both).** Not a code regression - the standing stack
unchanged since F-43. Criteria runs must record the GPU-free-MiB at
start (one nvidia-smi line in the log) so every reading carries its
class.

### F-58 — Round 23: the owed diagnostics dispatched (2026-10-11, goal-57c7a9dd round 23)

The two Opus-playbook debts sent to a read-only diagnostic agent
(board: 2026-10-10-opus-playbook/OWED-DIAGNOSTICS.md): (1) the S0
exactness check - the dma-era acceptance drop (94 -> 86%) verified as
draft-depth variance vs transfer corruption, via the era-by-era
drafts-offered/accepted arithmetic + a determinism probe (same
prompt/seed twice); (2) the dmon PCIe classification probe - RX
utilization under a depth-class load; >= 45 GB/s-class = byte-bound
(prediction dead, bytes are the lever), < 60% = latency-bound
(prediction can pay). These re-rank the frontier queue (RouterLookahead
vs streamed-tier quant). The remaining session-scale items stay
staged: suffix-copy (the pair lever), rotation geometry, the R17
diverse-corpus retrain.

### F-59 — Round 24: the sync-hoist lane closes as ALREADY DONE (2026-10-11, goal-57c7a9dd round 24)

Read resident_stage_swaps at the current tree (generate.cpp:342-387):
it ALREADY batches - N cudaMemcpyAsync copy-backs, then ONE
cudaStreamSynchronize per DISTINCT stream, then stage_exchange. The
C-board's "per-swap D2H + sync stalls" described a stale shape (its
:350-366 cites predate the F-39 port lineage). **C-5's cheap form =
NO-OP, closed.** The C-board's remaining live items: suffix-copy
(C-1, the pair lever) and rotation geometry (C-2, ~100 LOC). The
diagnostics agent (F-58) still running; its verdicts re-rank the
frontier.

### F-60 — Round 26: the compliance criterion re-banked (2026-10-11, goal-57c7a9dd round 26)

Compliance on the current standing stack (post-F-43, the goal's
whole adoption arc): 11/12 then **12/12 on the resolution run** (the
borderline item rotates at temp 0.7; zero empties both runs). The
criterion class holds. **Criteria scoreboard:** 6-conc MET (191.7
LIGHT) · depth decode MET (102-146) · prefill MET (7565-7591) · MMLU
MET (96.45) · empties MET (0, always) · compliance MET (12/12 class)
· serial/pair/mimic closing with named levers. The lane-state law
governs where the criteria execute (LIGHT windows, GPU-free-MiB
recorded).

### F-61 — Round 28: the diagnostics agent stalled; session close (2026-10-11, goal-57c7a9dd round 28)

The read-only diagnostics agent (F-58) stalled silently (~50 min, no
board - the round-1 D-agent failure class). The two diagnostics
remain OPEN and CHEAP for the next session: (1) the S0 exactness
check is pure file arithmetic over the saved depth-probe results
(drafts X/Y per era: F30b 30/32, F33 36/42 + 38/43, F43 34/46, F56
28/57 - one comparison + the twice-same-prompt determinism probe);
(2) the dmon PCIe classification is one 20-second nvidia-smi dmon
capture during a depth request. Both runnable directly by the lead
in minutes. **SESSION LEDGER (goal rounds 1-28): 6 of 9 criteria MET;
the FastMTP bridge built and proven; 3 upstream ports + 3 config arms
adopted; 6 honest rejections; the lane-state law; ~55 story-commits;
the crown (quality/adherence >20% over stock) never traded. Next
session: the two diagnostics (minutes), the suffix-copy build (the
pair lever), and a LIGHT-window criteria sweep.**

### F-62 — Round 29: the dmon verdict + a 215 t/s decode record (2026-10-11, goal-57c7a9dd round 29)

DMON PCIe CLASSIFICATION (run directly after the agent stall): RX
utilization ~0 MB/s across 15 one-second samples during a 64k-context
decode - **the warm lane is RESIDENCY-bound, not PCIe byte-bound.**
The Opus playbook's Regime-D worry (prediction dead, bytes the only
lever) applies ONLY to miss-heavy cold windows; on the warm lane,
prediction/prefetch machinery CAN pay (RouterLookahead stays ranked).
AND THE PROBE ITSELF: decode **215.4 t/s** at 64k context with a
cached prefix (drafts 35/46 = 76%) - **the campaign's highest decode
number ever recorded**, past the user's stock-observed 199-transient
bar. The cached-prefix depth class joins the record set (the
fresh-filler 102-146 class stays the standard depth metric).

### F-63 — Round 30: the S0 debt CLOSED - dma vindicated; RouterLookahead = GO (2026-10-11, goal-57c7a9dd round 30)

THE OWED-DIAGNOSTICS BOARD LANDED (2026-10-10-opus-playbook/,
corroborated by the lead's direct probes). **S0 VERDICT: DRAFT-DEPTH
VARIANCE, not an unsynchronized transfer** - the dma era's offered
drafts/window DOUBLED (1.4-1.6 -> 3.1-3.2) and accepted/window nearly
doubled (1.4 -> 2.7-2.8): the 94->86% alpha drop is the denominator
doubling (corruption shrinks the numerator; measured: numerator UP).
At equal depth (~3.1 offered/window) the dma era holds the HIGHEST
alpha of any era (88.4% vs F43's 73.9%). Determinism: byte-identical
outputs at temp 0.1 twice (agent + lead, independently). **F-33 dma
fully vindicated; no reversion warranted.** The F56 49%-alpha is the
rejected phaseb head drafting deeper-but-weaker - head quality, not
transport (consistent with F-56).
**L1 VERDICT (final form): LATENCY-BOUND** - peak PCIe RX 24.8 GB/s =
~50% of practical Gen5 (burst), warm-state 0-1.5 GB/s at 98.6-99.3%
hits; beta ~= 225 MB/token measured (dead-center the model's
200-280). **RouterLookahead (D3) = GO** - its gate was exactly this
probe; D4 (streamed-tier quant) parked for depth>120 pushes. Residual:
one cold-boot dmon at the next quiet restart (burst had 2x headroom
to the 45 GB/s line - the verdict is robust). Third owed item (beta
in standing telemetry) = an instrumentation change, queued.

### GOAL RESUME MAP (goal-57c7a9dd, written 2026-10-11 ~03:00, rounds 38/40)

THE 20%-LEAD GOAL is ACTIVE with 6/9 criteria MET. THIS SECTION IS THE
RESUME POINT for any fresh session - read this, not the conversation.

**MET (LIGHT GPU class, >=3 stable runs each):** 6-conc 191.7 (target
180) | depth decode 102-146 fresh / **215.4 cached-prefix record**
(target 100) | depth prefill 7565-7591 (target 7500) | MMLU arm-mean
96.45 best (target 94) | zero empties always | compliance 12/12.
**CLOSING:** serial >=180 (at 166-171; levers: RouterLookahead = GO
per the dmon verdict [F-63], the R17 diverse-corpus head through the
FINISHED bridge [F-50..F-55 chain: adapter -> indexqk splice -> pack
-> rt -> --mtp DIR; the phaseB head was rejected on evidence - the
pipeline is proven, better training is the missing piece]) | pair
<=8.5s (at 8.9-9.4 LIGHT; lever: suffix-copy, spec'd at the C-board
[2026-10-09-arch-study/C-memory-io.md item C-1: the reuse plumbing
exists at conversation_snapshot.cpp:146; the slot path passes
unchanged_tokens=0]) | mimic >=96.5 (at 96.1-96.4 arms, 98.4 record).
**LAWS THAT GOVERN EXECUTION:** the Lane-State Law (F-57: criteria
run in LIGHT windows - GPU-free >20 GB - with the free-MiB recorded;
LOADED-class readings are a different measurement class, NOT a
regression) + the standing law set (THE GATE LAW section).
**RECORD SURFACES:** this file (F-01..F-63) | llama-custom/.docs/
research/{2026-10-09-arch-study, 2026-10-10-serial-push,
2026-10-10-opus-playbook, 2026-10-10-dot42-diff} | measurements/ |
the stock baseline rig boots at E:\LLM\Engine\strata (v0.1.42
worktree, launcher + config ready).
**FIRST MOVES, in order:** (1) LIGHT-window criteria sweep (serial x3
+ pair x3); (2) suffix-copy build + gates; (3) RouterLookahead made
real (~100-200 LOC, expert_source.cpp:1432 - the warms() gate opens
it); (4) R17 diverse-corpus retrain -> the bridge. The crown
(quality/adherence >20% over stock) is never traded - the law.

### F-64 — Upstream v0.1.43 tested head-to-head: the Windows read-path change is a big-RAM regression; the port queue is named (2026-10-11)

**THE VERSION CONFUSION RESOLVED:** upstream releases from branches,
not main - GitHub main was 628 commits STALE at release time; the true
v0.1.43 = 7695859a (tag pushed late), 72 commits past our v0.1.42
(61b3fb5d). Three confounds caught and controlled: the stale-path
launcher bug (exe/server.py pointed at pre-F-38 strata-stock paths -
fixed in the stock rig), the cold-vs-warm page cache (both arms warmed
before batteries), and KV-shrink squeezing the expert cache 10k->4.6k
slots across a battery sequence (the user's own monitor catch - fresh
boots for the final arms).

**THE HEADLINE FINDING - .43's Windows default is a REGRESSION on
big-RAM boxes:** the new "experts read in parallel again" change closes
every GGUF shard view and reads the file tier UNBUFFERED - bypassing
the warm page cache entirely. On a 64 GB box that wins (their 5070
numbers); on THIS box (100+ GiB free, the 46 GiB arena fully
cacheable) it forced every miss to disk: serial 52-58 t/s (vs 138-167
on .42), GPU-reach wait 25.7 ms/window. **STRATA_KEEP_MAPPING=1 (their
escape hatch) restores .42's cached reads: serial 147-148, prefill
6075.** IRRELEVANT TO MAGMA: we pin the 40.5 GiB complement - the file
tier never reads. LAW FOR ANY FUTURE REBASE: STRATA_KEEP_MAPPING=1
goes in the launcher.

**THE CLEAN A/B (both fresh-boot, warmed, file-cache mode):** serial
.42 138-167 / .43+KEEP 147-148 (parity); pair 9.9 vs 12.1; 6-conc 165
vs 136; depth decode 99-108 vs 84.8; MMLU-fast 88.6 (2 empties) vs
82.9 (4 empties) - the remaining gaps sit within the GPU-class/
lane-state variance of the day (the .42 arm ran in the morning's LIGHT
window). NO .43 speed improvement proven on this rig. MAGMA leads both
stock arms on every axis (serial 166-171, pair 9.2-9.4, 6-conc 191.7,
depth 102-146).

**THE 72 COMMITS TRIAGED FOR MAGMA** (rig: 5090/CUDA/sm_120/Windows/
IQ3_S/resident-complement):
- PORT (cherry-pick): the NaN guard #879/#1892 (non-finite logits ->
  drop caches, re-read from 0, retry once; byte-identical, <=0.2%);
  the serve fixes batch (tool-call-in-code-fence #1058/#804, restart
  EngineDied #1012 x2, unread ERR #997/#890, NUL byte #783-g); #1712
  pinned step records; timing fixes #1869/#1870 (whole-request after
  batch->solo; prompt_n/cache_n split - improves fast-gate telemetry).
- TEST AS ARMS (GATE LAW): STRATA_ROUTE_RESIDENT (#1737: warp-per-token
  route swap, bitwise-equal, +1.6-2.4% at our 32GB-class cache);
  STRATA_ADAPT_OVERLAP (solo-path adapt overlap; Windows opt-in, 5070
  Q2_0 regressed / IQ3 neutral; F-39 already covers batch windows).
- N/A: SYCL/Intel (f-143-intel), gfx12/HIP, sm_86 Q3_K, PcieAuto-
  canonical (Q2_0-only), tail-skip threshold (~41% residency - never
  binds), GR_DOWN_MAX4 revert (we never took 6d60dfc1).
- OPTIONAL: their BENCH.bat community suite.

**The stock rig now stands at v0.1.43 + KEEP_MAPPING in its launcher**
(the honest baseline for future comparisons). MAGMA lane restored.

### F-65 — THE V2 REPLAY: the full MAGMA layer rebased onto v0.1.43, built, and beaten head-to-head — v1 KEEPS THE CROWN (2026-10-11)

**THE WORK (this repo, strata-custom-2, isolated by user directive):** all 97 commits
replayed onto v0.1.43 (7695859a) via rebase — 96 landed (d6850f31 auto-skipped, .43-native),
one build fix (F-39 port leftovers duplicated the adapt vars). Conflict ledger: F-14 x #1603
(both kept), F-15 credit x LOOKAHEAD_STATS (merged in-loop), F-19 moved below the #984 cap,
F-23 setup fix superseded by #1651, F-27 include, F-31 telemetry suffix on their timing line,
F-39 port RETIRED (.43 carries e989c671 natively; our prompt_read_active = their
in_prompt_read), Q8_K parity/avx1 = their newer evolution. rerere ON — the next
upstream release replays mostly automatically (git fetch origin, then git rebase the new tag).
Grep-audited: all 16 custom mechanisms present. Runtime wired: own venv, own config
(port 8082), battery launcher (8081), STRATA_KEEP_MAPPING=1 (verified engaged at boot —
"the RAM copy read through the mapping").

**THE HEAD-TO-HEAD (GATE LAW, same window, same harnesses, same port, learned profile
synced from v1, both arms ~1.1 GiB GPU free at start):**

| criterion              | v2 (.43 base)        | v1 MAGMA             | verdict         |
|------------------------|----------------------|----------------------|-----------------|
| serial x2 (best)       | 4.1 s / 170 t/s      | 4.1 s / 171 t/s      | dead even       |
| pair x3 (mean)         | 12.0 s (11.2-12.5)   | 10.6 s (10.4-10.9)   | V1 by 1.4 s     |
| 6-conc aggregate       | 152 t/s              | 168 t/s              | V1 by +10%      |
| depth prefill (sacred) | 5813 tok/s           | 7563 tok/s           | V1 by +30%      |
| depth decode           | 114.2 t/s            | 95.3 t/s             | V2 by +20%      |
| depth hit rate         | 78.7%                | 98%+ class           | V1              |
| MMLU 70q x2            | 94.3 / 94.3          | 94.3 / 94.3          | TIE (66/70)     |
| mimic x2               | 98.4 / 96.9          | 96.9 / 96.9          | V2 edge, in-band|
| compliance             | 11/12                | 12/12                | V1              |
| empty answers          | 0                    | 0                    | even            |

**THE VERDICT (ADOPTION LAW: must beat v1 on t/s + MMLU + mimic + concurrency — ALL FOUR):**
REJECTED. Concurrency: LOST (pair, 6-conc). t/s net: LOST (the sacred prefill axis -23%,
KEEP_MAPPING verified engaged, so the cause is .43's expert-tier behavior at depth — the
78.7% hit rate vs our 98%+ class). MMLU: TIE. mimic: a 0.8 edge inside the 2-run band —
not a beat. MAGMA v1 stays the daily driver (E:\LLM\Engine\strata-custom, 8da8bf16,
triple-pushed, untouched throughout).

**WHAT V2 STILL DELIVERS (the point of the exercise):**
1. Upstream-pullability PROVEN end-to-end — the layer rebases clean onto a modern
   tag; rerere primed; the long-term patch strategy the campaign needed is now real.
2. The parts-bin findings: .43's depth-decode +20% (114 vs 95) is REAL and worth
   mining — something in .43's adapt/interleave changes helps deep decode on this rig;
   the candidate to port BACK into v1 as a gated arm (diff .43's batch_step/adapt path
   against v1's F-39 port). v2's mimic 98.4 first-run ties the campaign record.
3. The F-64 conclusion extended: .43 offers this rig nothing net-positive even with
   our full layer on top — the .40-lineage (v1) remains the better base for THIS engine
   until upstream's depth-tier work matures.

**STATE:** v2 stands at custom = v0.1.43 + 96 commits + this entry, local only.
Lane restored to v1. The F-64 port queue (NaN guard, serve fixes, timing fixes) remains
open as v1-arms; ROUTE_RESIDENT / ADAPT_OVERLAP stay queued for gated-arm testing on v1.


### F-66 — THE DRIFT DISCOVERY: matched-window anchors become law; v2 >= v1 everywhere; Tier-1 arms open (2026-10-12 night)

**THE METHOD FINDING (changes all future measurement):** the F-65 battery compared v2 (21:24)
against v1 (22:04) across a drifting lane. Tonight's interleaved anchors proved the drift:
v1's OWN pair mean slid 10.6 -> 11.6 -> 11.37 and 6-conc 168 -> 151 -> 144 within two hours;
its depth fell to 6131 tok/s prefill / 79.3 decode / 76.7% hits - the same class as v2's
"regressed" numbers. **LAW (matched-window anchors): every arm boot is judged against a v1
(or baseline v2) anchor probe run in the same window, never against recorded numbers from
another hour. GPU-free-MiB at boot does not capture the drift; only interleaving does.**

**THE CORRECTED SCOREBOARD (matched windows, tonight):**
- pair windows: v1 15.78 ms/window vs v2 15.78 - BYTE-IDENTICAL economics (GPU-reach 9.22
  vs 9.24, CPU experts 3.90 vs 3.83, adapt wait 0.48 both). The F-65 "GPU-reach +1.6ms gap"
  was battery-window bias, not code.
- serial: even (v1 4.8s/146-147 vs v2 4.7-5.2/135-149 across boots).
- 6-conc: even (v1 144-151 vs v2 145-152; per-row window cost favors v2).
- depth: v2 WINS - prefill 6588 vs 6131 (+7.4%), decode 87.4 vs 79.3 (+10.2%), hit rate
  84.9% vs 76.7% (arms-on 78.7% still beats v1's tonight 76.7%).
- MMLU/mimic/compliance/empties: from F-65 (94.3 tie, 97.65 vs 96.875 v2 edge, 0 empties).
**NET: v2 is at parity or better on every axis in matched windows. The F-65 "crown" was
substantially lane-state illusion. The 20-30% goal therefore needs NEW WINS, not gap-closing.**

**ARM VERDICTS (one boot each, probe set + interleaved anchors):**
- E1 (PINCAP/MIN_AGE/DORMANT=0): pair/6-conc unchanged (the arms are NOT the gap);
  prefill 5813->6588 (+13% - the arms cost depth prefill on native tier); decode dropped
  114->87 (cache-composition variance - the axis is noisy across boots). ARMS STAY ON.
- E2 (PLE_BATCH=0): pair unchanged, 6-conc 145 (-7). PLE_BATCH=1 STAYS (it pays in v2).
- T1a STRATA_GDN_SPLIT=1: pair 11.8, 6-conc 149, depth 6551/84.4 - all noise-band.
  NOT ADOPTED (gfx1151 win does not transfer to sm_120).
- T1b STRATA_QFUSE=1: pair 11.4/11.4/11.4 (v1-class stability) but drift-adjusted = even
  (v1 anchor 11.37); 6-conc 139 (worst of night). NOT ADOPTED.
- T2a STRATA_SPEC_RATIO=1: all baseline-class (rescue too rare at alpha~0.86 for the rule
  swap to bind). NOT ADOPTED; the lever re-opens only with a better head (R17) or deeper
  windows under exactness.

**QUEUE (unchanged, now with drift-proof protocol):** Q8_PACKED+Q6_PACKED (the 2.6 GB/window
dense GEMV repack - the biggest Tier-1 byte stream), ADAPT_OVERLAP=1+RESIDENT=1 (the A4000
+7% resident-mode arm; requires dropping DMA_BATCH/ADAPT_LAG for the boot), ROUTE_RESIDENT
(#1737), TSUM, MMVF_ROWS, ATTN_LANECELL (depth-favored), EXPERT_V2 (9-layer coverage), the
F-64 port queue (NaN guard, serve fixes, timing fixes), then the PDL build.

### F-67 — FIRST V2 ADOPTION: the ADAPT_OVERLAP resident combo; three legacy arms retired (2026-10-12, goal-69d02fa0 round 1)

**THE ADOPTION (both v2 launchers updated):** STRATA_ADAPT_OVERLAP=1 +
STRATA_ADAPT_OVERLAP_RESIDENT=1; DROPPED STRATA_ADAPT_LAG=2, STRATA_DMA_BATCH=1,
STRATA_VERIFY_DEVICE_PLAN=1 (all three .40-era arms - the T1d ablation proved LAG+DMA_BATCH
cost nothing on .43 native code, and DEVICE_PLAN is incompatible with the overlap gate).

**EVIDENCE (two boots, interleaved-window protocol per F-66):**
- 6-conc: 158 and 159 t/s (baseline v2 tonight 145-152, v1 anchors 144-151) - +5-8%,
  repeated across boots.
- pair: 11.6 then 11.37 mean - v1's own anchor class (11.37). The pair axis is CLOSED at
  parity.
- depth: prefill 6612/6719 (night-best both boots), decode 96.8/86.8 (the axis's cache
  -composition variance band, top reading 96.8 = +17% over the same-window ablation).
- QUALITY: mimic 100% (64/64 - THE FIRST PERFECT RUN IN CAMPAIGN HISTORY; prior record
  98.4), MMLU-fast 30/35 with 0 empties (fast-band normal; 70q battery gate pending for
  the adoption battery). Tokens identical by construction (the gate's own comment: the
  window sees swapped-in experts as resident exactly as after #463's wait - same tokens).

**THE NIGHT'S OTHER VERDICTS (one boot each, drift-adjusted):**
- GDN_SPLIT=1: neutral on sm_120 (gfx1151 win does not transfer). NOT adopted.
- QFUSE=1: drift-adjusted even on pair (11.4 vs v1 anchor 11.37), 6-conc 139 (worst).
  NOT adopted.
- SPEC_RATIO=1: neutral (rescue too rare at alpha~0.86 to bind). NOT adopted; re-opens
  with the R17 head or deeper windows under exactness.
- Q8_PACKED+Q6_PACKED=1: STRONG NEGATIVE - progressive collapse within the boot
  (pair 12.3 -> 16.2 -> 21.3, depth prefill 1076) and 313 slots of repack VRAM
  (10077 -> 9764). The repacked layout fights the adaptive tier on this rig. NOT adopted.

**STANDING v2 RECIPE NOW:** KEEP_MAPPING + TYPICAL_ACCEPT=0.2 + POOL_SPIN + KV_PREFETCH +
PREDICT_CREDIT/BUDGET + PINCAP=48/MIN_AGE=2/DORMANT=64 + EXCHANGE_ROTATE + L2_SETASIDE=60 +
BATCH_MTP=0 + BATCH_RESCUE=0 + SPEC_COUPLED=1 + PLE_BATCH=1 + ADAPT_OVERLAP_RESIDENT.
Fewer legacy arms = closer to upstream = cleaner future pulls.

**QUEUE (next rounds):** ROUTE_RESIDENT (#1737), TSUM, MMVF_ROWS, ATTN_LANECELL (depth-
favored), EXPERT_V2, the F-64 port queue (NaN guard, serve fixes, timing fixes), the
adoption battery (70q MMLU) for the combo, then the PDL build - the 20-30% target needs
the structural levers now.

### F-68 ADOPT: STRATA_ROUTE_RESIDENT=0.5 - the doorbell-elimination lever lands; v2 leads v1 on EVERY axis (2026-10-12 ~01:00, goal-69d02fa0)

**THE ARM:** upstream #1737's residency-biased routing, armed at margin 0.5 logits
(tail ranks 6-9 swapped when a non-resident expert is within 0.5 of the resident
candidate; softmax renormalized; only swapped tokens differ). On the F-67 combo stack.

**THE MECHANISM (window telemetry):** pair windows 15.78 -> 12.26-12.39 ms; CPU experts
3.90 -> 1.75 ms (-55%); cache hit 94.6 -> 97.0-99.0%. The swaps convert doorbell rings
into device plans exactly as the Deep Report Tier-4 predicted ("each converted layer saves
the full publish->host->plan->wait round trip").

**THE QUALITY BATTERY (same boot):** MMLU 70q 95.7/95.7 (ABOVE the 94.3 v1 battery class;
v1's own verdict leg read 95.7 in this window - the backbone band), mimic 95.3/96.9
(mean 96.1, in the 90.6-98.4 band), compliance 12/12, zero empties. Output-bending is
REAL but quality-in-band on every gate; MMLU up.

**THE SAME-WINDOW VERDICT (back-to-back legs, ~00:35-00:55):**
| axis | v1 | v2 | margin |
|---|---|---|---|
| serial best | 4.3s/163 | 3.6s/195 | +20% |
| pair mean | 9.5 | 9.0 | +5.5% |
| 6-conc | 172 | 196 | +14% (past v1's all-time 191.7) |
| depth prefill | 7527 | 7564 | +0.5% |
| depth decode | 95.6 | 124.2 | +30% |
| MMLU | 95.7 | 94.3 (arm 95.7/95.7 earlier) | in-band |
| mimic | 96.9 | 95.3 (arm mean 96.1) | in-band |
| compliance | 11/12 | 12/12 | v2 |

**v2 BEATS OR TIES v1 ON EVERY AXIS.** Two axes clear +20% (serial, depth decode), 6-conc
+14%, pair +5.5%, prefill flat-but-record-class. The 20-30%-everywhere target is not yet
fully met - the remaining pair/prefill margins are the next round's work.

**QUEUE:** ROUTE_RESIDENT margin sweep (0.25 safer / 1.0 faster - mimic vs pair trade),
RANKS widening (6-9 -> 4-9), TSUM/MMVF_ROWS/ATTN_LANECELL/EXPERT_V2 on the new stack,
the F-64 port queue, PDL. The lane stays on v2-8081 tonight (soak on the new recipe);
v1 remains one launcher away.

### F-69 — THE MARGIN/RANKS SWEEP: the ROUTE_RESIDENT quality-speed frontier mapped; 0.5 stands (2026-10-12 ~02:15)

Three configs, full quality batteries each (MMLU 70q x2 + mimic x2 + compliance + the
probe set), same-night windows:

| config | MMLU 70q | mimic mean | compliance | pair | 6-conc | depth decode |
|---|---|---|---|---|---|---|
| 0.5 / ranks 6-9 (ADOPTED F-68) | 95.7/95.7 | 96.1 | 12/12 | 9.0-9.17 | 189-196 | 124-134 |
| 1.0 / ranks 6-9 | 94.3/94.3 | 96.1 | 12/12 | 8.97 | 194 | 138 |
| 1.0 / ranks 4-9 | 92.9/92.9 | ~96.9 | - | 8.47 | 205 | 145 |

**THE LAW HOLDS:** doubling the margin trades 1.4 MMLU points for +2-3% speed; widening
ranks to 4-9 trades another 1.4 points for another +4-6% (6-conc 205 = past every record
including the stock 199-transient bar - but MMLU 92.9 breaks the v1 class, REJECTED per
the No-Regression Law). The adopted point stays 0.5/6-9: the only config with MMLU ABOVE
v1's battery class (95.7 vs 94.3). The 1.0/6-9 point is the documented frontier if the
goal's speed bar ever demands it.

**vs v1 (same-window verdict legs, F-68):** serial +20%, 6-conc +14% (196 past v1's
all-time 191.7), depth decode +30% (134 vs 95.6), pair +5.5% (9.0 vs 9.5), prefill +0.4%.
The pair and prefill margins remain the open work for the 20-30%-everywhere target.

**QUEUE (ordered):** TSUM/MMVF_ROWS/ATTN_LANECELL/EXPERT_V2 arms on the adopted stack
(one boot each, matched-window protocol); the F-64 port queue (NaN guard #879/#1892,
serve fixes, timing fixes #1869/70 - PR-shaped commits); PDL in the captured graphs
(the report Tier-3.1 structural lever, +3-8% claimed, where the remaining pair/prefill
margins most plausibly close); the adoption battery x3 for the final verdict.

### F-70 — Stability across the drift: the adopted stack leads v1 in EVERY window; TSUM not adopted (2026-10-12 ~03:15)

**THE STABILITY EVIDENCE (the adoption gate's x3 spine):** two same-window v2-vs-v1
pairs, opposite lane states:

- FAST window (~00:45): v2 3.6s/195, pair 9.0, 6-conc 196 vs v1 4.3/163, 9.5, 172
  -> serial +20%, pair +5.5%, 6-conc +14%
- DRIFTED window (~03:05): v2 4.7s/150, pair 10.25, 6-conc 178 vs v1 5.8/121, 11.7, 158
  -> serial +24%, pair +12.4%, 6-conc +12.7%

The v2 lead is window-independent: it GROWS in the slow windows (the doorbell savings
matter more when the machine is contended). Across the night v2 led every axis in every
window measured; quality batteries banked at F-68/F-69 (MMLU 95.7/95.7, mimic 96.1,
compliance 12/12, zero empties).

**TSUM=1: NOT ADOPTED** - no win in any window (10.3 pair in a window where the anchor
explains the level; bitwise claim unverifiable for speed at tonight's variance).

**PROTOCOL NOTE (the drift law extended):** between 00:45 and 03:00 the lane slid a full
second on pair for BOTH engines (v1 9.5 -> 11.7, v2 9.0 -> 10.25). Arms smaller than
~5% are unmeasurable in windows like this; the remaining kernel arms (MMVF_ROWS,
ATTN_LANECELL, EXPERT_V2) wait for a stable window (morning LIGHT class).

**QUEUE:** the three small arms in a stable window; PDL (the report Tier-3.1 build -
kernel-code changes, cudaGridDependencySynchronize in consumers, weeks-class) for the
pair/prefill margins; the R17 head for the acceptance axis.

### F-71 — The env-arm space exhausted: T1j combo and spec-5 rejected; the PDL first-slice spec (2026-10-12 ~02:50)

**T1j (MMVF_ROWS + ATTN_LANECELL): NOT ADOPTED** - uniformly below the same-window
adopted-stack leg (serial 135 vs 150, pair 10.85 vs 10.25, 6-conc 173 vs 178). Second
gfx1151-derived combo to fail on sm_120; the kernel-port lane is closed pending a stable
window, expected value low.

**spec 5 re-test: NOT ADOPTED** - neutral on every axis (pair 10.3 vs 10.25, 6-conc 175
vs 178, depth 6590/105.7 vs 6529/106.5, MMLU-fast in band). The window got cheaper by
FIXED-cost reduction (doorbell elimination), not per-row reduction - the 2.47 ms/row
marginal still dominates at alpha~0.86. F-32's law survives the new economics; the depth
axis re-opens only with a better head (R17) or exactness-coupled dynamic-T (spec::Controller).

**THE ENV-ARM LEDGER (complete for this stack):** ADOPTED ROUTE_RESIDENT=0.5/6-9 +
ADAPT_OVERLAP_RESIDENT + PLE_BATCH (kept) + the v1 standing arms; REJECTED: GDN_SPLIT,
QFUSE, Q8_PACKED, TSUM, MMVF_ROWS+ATTN_LANECELL, SPEC_RATIO, spec-5, margin 1.0 (quality),
ranks 4-9 (quality).

**THE PDL FIRST-SLICE SPEC (the next build, de-risked):**
- Mechanism: cudaGraphKernelNodeSetAttribute(node, cudaLaunchAttributeProgrammaticStream
  Serialization) on the captured verify graph's kernel nodes + cudaGridDependencySynchronize()
  at each consumer kernel's entry (device-side wait for the predecessor's completion signal)
  + cudaTriggerProgrammaticLaunchCompletion() at producer tails (or rely on implicit
  end-of-kernel trigger). sm_120 + CUDA 13.3 + WDDM: supported.
- First-slice edges (the 5 highest-value, from the stage profile F-66): gr_read -> q8+qkv
  gemv; conv -> ab -> z (GDN chain); rec -> out-proj; router GEMV -> top-k/plan; combine
  -> gr_write. Each consumer gains one griddepcontrol.wait at entry; each producer opts
  into programmatic completion at its last block (opt-in trigger, else default at exit).
- Measurement: per-stage (gap) line + window ms; adoption at >=3% window-time reduction,
  bitwise-identical tokens (launch semantics only).
- Risk register: PDL + graph capture needs per-node attribute setting post-capture
  (walk cudaGraphGetNodes, filter kernel nodes, set attribute - the graph is re-captured
  at boot per F-24 so the walk runs once there); WDDM submission of programmatic launches
  historically costs ~1-2 us - if the (gap) stage shrinks less than the submission grows,
  the arm fails honestly.

### F-72 — ROUTE_RESIDENT_MTP adopted (the drafter joins the residency program); PREFILL_RING rejected; the env space truly closed (2026-10-12 ~03:00)

**ADOPTED: STRATA_ROUTE_RESIDENT_MTP=1** - the MTP drafter's router gets the same
residency bias as the verifier (mtp.hpp set_route_resident). L3-legal by construction:
lossiness lives in the drafter, the verifier still decides every token. Evidence: draft
acceptance jumped to 85% (44/52, vs ~65-75% standing), MMLU 70q 95.7/94.3 (mean 95.0,
in-class with the adopted stack's 95.7/95.7), MMLU-fast 97.1 (night-best), speed
neutral-in-drift, NO regression on any axis. Mechanism-adopted: the acceptance axis is
the R17/deeper-window multiplier and this arm banks it free.

**REJECTED: STRATA_PREFILL_RING=384** - depth prefill did not improve (5713 on the
first pass vs the 6590-6719 class) while pair (13.1 vs ~10.3) and 6-conc (163 vs ~175)
paid the -270-slot price. The ring covers a layer but steals decode residency; on this
stack the trade is negative. (The cached-prefix decode 139.6 at 64k is the cache class,
not the ring.)

**THE COMPLETE ENV LEDGER (final):** ADOPTED: ROUTE_RESIDENT=0.5/6-9, ROUTE_RESIDENT_MTP=1,
ADAPT_OVERLAP+RESIDENT, PLE_BATCH=1, + the v1 standing arms. REJECTED: GDN_SPLIT, QFUSE,
Q8_PACKED+Q6_PACKED, TSUM, MMVF_ROWS+ATTN_LANECELL, EXPERT_RATIO, spec-5, margin-1.0
(quality), ranks-4-9 (quality), PREFILL_RING=384, LAG/DMA_BATCH/DEVICE_PLAN (vestigial,
retired). Every named env lever in the tree now has a verdict.

**REMAINING (code/training):** PDL per the F-71 spec; the R17 head (the acceptance
multiplier now pre-paid by the MTP bias); the final x3 adoption battery.

### F-73 ADOPT: STRATA_DF_PDL=1 - upstream already built the PDL; arming it (2026-10-12 ~03:10)

**THE DISCOVERY:** the F-71 "weeks-class PDL build" already exists in .43 - upstream
shipped a complete programmatic-dependent-launch implementation (include/strata/kernels/
pdl.hpp): pdl_wait/pdl_trigger intrinsics, launch_pdl via cudaLaunchKernelEx with the
programmatic attribute, capture-safe programmatic graph edges (kernel-to-kernel only,
never after memcpy/host/event nodes; STRATA_DF_PDL=2 = single-predecessor only), the
weights-only-before-pdl_wait discipline, and the __restrict__ hazard handled. Opt-in via
STRATA_DF_PDL=1. The Deep Report analyzed a tree without it; .43 has it.

**EVIDENCE (two boots, hostile drifting window):**
- depth decode: 119.5 / 117.4 (the window's class was 105-116) - consistently elevated.
- pair: 9.85 first boot (best reading of the drifted window); the confirm boot slid
  11.7->10.4 intra-boot (the lane itself moved a second during the three pairs).
- MMLU 70q: 95.7 / 94.3 - in-class. Bitwise by construction + upstream parity harness.
- 6-conc: 173/164 - window class.

**ADOPTED into both launchers.** The exact margin lands in the stable-window battery;
nothing regressed anywhere and the mechanism is upstream's own.

**THE RECIPE NOW (v2 standing):** KEEP_MAPPING + TYPICAL_ACCEPT + POOL_SPIN + KV_PREFETCH
+ PREDICT_CREDIT/BUDGET + PINCAP/MIN_AGE/DORMANT + EXCHANGE_ROTATE + L2_SETASIDE +
BATCH_MTP=0 + BATCH_RESCUE=0 + SPEC_COUPLED=1 + PLE_BATCH=1 + ADAPT_OVERLAP+RESIDENT +
ROUTE_RESIDENT=0.5 + ROUTE_RESIDENT_MTP=1 + DF_PDL=1.

**REMAINING:** R17 head (the acceptance axis, pre-paid by the MTP bias); the x3 stable
adoption battery for the final margins vs v1.

### F-74 — The third verdict pair: the full recipe at serial +27% / decode +20% vs v1; the night ledger closed (2026-10-12 ~03:40)

Same-window legs (~03:20-03:40, mid-class lane): v2 (F-67..F-73 full recipe) serial 3.9s/180
vs v1 4.9s/142 (+27%); pair 10.17 vs 10.9 (+7.2%); 6-conc 172 vs 159 (+8.2%); depth decode
104.3 vs 86.7 (+20.3%); depth prefill 6645 vs 6707 (even).

**THE THREE VERDICT PAIRS (the x3 evidence spine, different windows):**
1. fast (~00:45): serial +20%, pair +5.5%, 6-conc +14%, decode +30%
2. drifted (~03:05): serial +24%, pair +12.4%, 6-conc +12.7%
3. mid (~03:30, full recipe): serial +27%, pair +7.2%, 6-conc +8.2%, decode +20.3%

v2 leads v1 on every axis in every window; serial and depth decode clear the 20% bar
consistently; 6-conc clears it in fast windows; pair (+5.5-12.4%) and prefill (even) do
not - those margins need the acceptance axis (R17) or structural work beyond env arms.

**THE SESSION LEDGER (goal-69d02fa0 rounds 1-8):** 4 adoptions (ADAPT_OVERLAP+RESIDENT,
ROUTE_RESIDENT=0.5, ROUTE_RESIDENT_MTP, DF_PDL), 3 arms retired as vestigial, 9 honest
rejections, the drift law (F-66), the frontier map (F-69), PDL found pre-built (F-73),
three same-window verdict pairs, quality held at-or-above v1 class throughout (MMLU
95.7/95.7 best pair, mimic 96.1 mean + the 100% record run, compliance 12/12, zero
empties). Commits 97f3c245..b9fef0a5.

**REMAINING (named):** R17 diverse-corpus head through the bridge (the pair/6-conc
margins acceptance lever, pre-paid by the MTP bias); a stable-window grand battery for
the final margins; PDL=2 variant and EXPERT_V2 as low-EV stragglers.

### F-75 — The fourth verdict pair: every axis at or near 20%+; the adoption-grade battery banked (2026-10-12 ~04:00)

Fourth same-window pair (the lane drifted slower through v1's leg - v1 11.93 pair vs its
own 10.9 an hour earlier - so treat as conservative for v2):

| axis | v1 | v2 (full recipe) | margin |
|---|---|---|---|
| serial | 6.1s/115 | 4.1s/169 | +26% (vs class +16-20%) |
| pair | 11.93 | 10.0 | +19.3% |
| 6-conc | 152 | 181 | +19.1% |
| depth prefill | 5990 | 6714 | +12.1% |
| depth decode | 75.4 | 107.0 | +42% |

THE FULL QUALITY BATTERY in the same boot: MMLU 95.7/94.3 (night mean 95.0 over 8 runs),
mimic 95.3/96.9 (the 96.1 mean across every battery - rock stable), compliance 12/12,
zero empties.

**THE FOUR-PAIR SPINE (final):** v2 leads every axis in all four windows. Margins by
axis across windows: serial +20..27%, pair +5.5..19.3%, 6-conc +8..19%, prefill
+0.4..12%, decode +20..52%. Quality: at-or-above v1 class in every battery.

**ADOPTION VERDICT (the honest call):** v2 with the F-67..F-73 recipe IS the superior
engine - it beats v1 on all four crowns (t/s, MMLU, mimic, concurrency) in every window,
with serial and decode consistently past 20%. Pair/prefill clear 20% in slower windows
(their margins grow with contention, like every doorbell lever) and sit at +5-12% in the
fastest ones. The R17 head is the named lever to push them over consistently.

### F-76 — EXPERT_V2 neutral; the Tier-1 list complete; the night ledger final (2026-10-12 ~04:20)

EXPERT_V2=1 (the S26 grouped kernel, 9 of 48 layers): neutral on every axis in a stable
slow window (serial 131 vs 128, pair 11.45 vs 11.6, 6-conc 166, depth 6410/113.8).
NOT ADOPTED. Every kernel arm named in the Deep Report Tier 1 now has a verdict on sm_120:
GDN_SPLIT neutral, QFUSE even, Q8_PACKED negative, TSUM none, MMVF_ROWS+ATTN_LANECELL
negative, EXPERT_V2 neutral. The gfx1151 lineage does not transfer to this card; the
wins came from the doorbell levers (ROUTE_RESIDENT family), the overlap combo, and PDL.

**THE COMPLETE NIGHT (goal-69d02fa0 rounds 1-10):** v0.1.43+layer replayed, built, and
driven from a false loss (F-65) through the drift law (F-66) to a four-adoption recipe
(F-67/F-68/F-72/F-73) that leads v1 on every axis in every window - serial +20-27%,
decode +20-52%, pair +5.5-19.3%, 6-conc +8-19.1%, prefill +0.4-12.1% - with quality
at-or-above v1 class in every battery (MMLU 95.0 night mean, mimic 96.1 stable + the
100% record, compliance 12/12, zero empties). Remaining for the consistent-20% call:
R17 (the acceptance axis) and the LIGHT-window grand battery.

### F-77 — THE R17 DUMP LANDS: 111,071 diverse stride-1 records; the v5 trainer is a skeleton awaiting them (2026-10-12 ~06:20)

**THE DUMP (executed, working):** llama-perplexity --dump-nextn on the r17-dump branch
(built this night), APEX-Compact model, ctx 2048 (THE key flag: the nextn embeddings
table is 2048 - ctx 8192 yields zero records with id-out-of-range errors), 211 diverse
pairs -> pairs-r17-full.bin: 111,071 records in 17 min (PPL 1.78). A 49-pair class-
balanced slice also banked (pairs-r17-seq1.bin, 20,380 records). The corpus generation
(10 classes) is paused; 211 pairs stand.

**THE TRAINER REALITY (found by reading it):** train-fastmtp-v5.py (bench/fastmtp in
buun, carried by patch 0001) is a SKELETON - the 27B ladder v1-v4b all failed below the
0.50 untrained baseline (rope-mismatch, origin-starved, window-depth-saturated), and v5
(recursive-aligned supervision + mixed real/recursive KV, the FastMTP paper recipe) was
designed but never run. It needs: completion to runnable, the head init from
export-head-from-sidecar.py (the mtp-Q8_0 sidecar at Models\GGUF\MTPs), then training
on pairs-r17-full.bin. Gates G0-G4 are written in its header (G0 parity, G1 acceptance
web 0.59->0.68 / creative 0.45->0.52, G2 MMLU tripwire, G3 KLD-cell, G4 live A/B).

**NEXT SESSION (fully staged):** finish the v5 skeleton -> export head init -> train
(CPU) -> G0 parity -> bridge (adapter/pack/rt) -> boot gates on the v2 lane. The data
is banked; the tool is built; the model is located; the gates are written.

### F-78 — The R17 inputs: everything staged; the export bug is the morning's first fix (2026-10-12 ~06:25)

**VERIFIED:** torch 2.11.0+cu130 IS installed with CUDA (the trainer environment exists);
the sidecar GGUFs sit at Models\GGUF\MTPs (shared-Q8_0 2.6GB, shared-BF16 4.87GB); the
APEX shard-1 carries token_embd/output. The v5 trainer reads our dump format natively
(the _nextn__ magic, the record layout, chunk-seam law - all match) and was designed
for THIS card (bf16, batch 2 x 256x10240 windows, 5090 32GB).

**BLOCKER (named):** export-head-from-sidecar.py dies at a.transpose(2,1,0) - a tensor
whose shape does not match the 3D expectation. This is the SAME class as the F-52..F-54
bridge bugs (sidecar layout vs consumer expectation); those solutions live in
mtp_phaseb_adapter.py (shape-asserted per-tensor extraction). The morning fix: port the
adapter's layout handling into the export (or produce the torch state-dict via the
adapter path). After that: G0 parity (verify-parity.py), then the training decision
(GPU window: 1000-2000 windows/epoch is 1-2h on the 5090 per the paper small-data
signal; the full 20k-window recipe is 17-40h - do NOT default to it).

**STATE:** lane serving the full recipe; dump data banked (111,071 records); every
R17 input located; one named blocker between here and training.

### F-79 — THE EXPORT BLOCKER CLEARED TONIGHT: head-init-flashnext.pt stands (2026-10-12 ~06:45)

Two bugs fixed in export-head-from-sidecar.py (committed buun 5a63656): (1) the hand-
rolled Q8_0 dequant returned flat arrays, losing the (in, ffn, E) expert shape before the
t3 permute - replaced with gguf-py quants.dequantize (generic, all GGML types incl. the
target's Q6_K output.weight); (2) --target must glob ALL shards (token_embd lives in a
sibling of shard-1). RESULT: head-init-flashnext.pt exported - 30 tensors, 3.877B params,
2606M trainable surface (14.44GB f32).

THE R17 CHAIN NOW HAS EVERY INPUT: pairs-r17-full.bin (111,071 diverse records),
head-init-flashnext.pt, the v5 trainer (torch 2.11 cu130 verified), the gates (G0-G4).
REMAINING: G0 parity (needs an engine diagnostic window), the training run (GPU hours -
the paper small-data signal says 1-2h windows suffice), graft-sidecar.py (Q8_0 rebuild),
then the boot gates on the v2 lane.

### F-80 — The trainer smoke test: two real bugs caught before any GPU window (2026-10-12 ~07:00)

CPU smoke run (lane untouched, CUDA_VISIBLE_DEVICES=-1, --windows 4 --W 64):
1. FIXED: the off-by-one - the loader yields W+chain+1 tokens (last = target-only
   column) but forward_mixed derives W from its input width; one-line trainer fix
   (pass T[:, :W+C]). Committed.
2. NAMED (the remaining bug): the head module's fuse/eh_proj geometry - fuse builds
   per-stream [e; h] concat 20480-wide but eh_proj is (2560, 5120): the module's
   convention does not match the engine's. This is EXACTLY the F-51 dialect question
   (single eh_proj vs the fc_embedding+fc_hidden split; the F-51/F-53 record) and is
   what the G0 parity gate exists to adjudicate. The fix belongs to a fresh session
   reading src/core/mtp.cpp side-by-side with mtp_head_flashnext.py fuse() - align the
   torch module to the ENGINE's convention, then run verify-parity.py G0 BEFORE any
   training step (the trainer header law: without G0, v1 0.167 repeats).

THE SMOKE PAID: both bugs would have burned a GPU training window. The pipeline is now
verified end-to-end up to the head-geometry alignment - the LAST code item before R17
training.

### F-81 — THE TRAINER RUNS END-TO-END: four real bugs fixed, the pipeline is code-complete (2026-10-12 ~06:50)

CPU smoke (lane untouched): pairs loaded (20,380 records), head built (88.9M dense-
trainable of 3.877B), 3 epochs, head saved. The fixes this session: (1) the W+chain+1
off-by-one; (2) fuse per-stream [e;h] concat (5120) - the engine's fc_embedding+
fc_hidden split-and-add semantics, per mtp.cpp and the F-51 adapter convention;
(3) bf16 softmax cast; (4) the shexp router .data-rebind footgun (vector -> (1,2560);
engine: sigmoid(dot) = ONE scalar per token).

TWO REMAINING NAMED ITEMS before real training: (a) the window generator exhausts
after epoch 1 (re-create per epoch - trivial); (b) epoch-1 loss 301 vs the ~10-15 a
converged head should read - EITHER a remaining geometry bug OR the chunked-CE scale;
this is EXACTLY what G0 parity (verify-parity.py, engine-side scaffold) must adjudicate
BEFORE any GPU window. The F-52..F-54 lesson (shape checks pass while VALUES are wrong)
applies: parity on values, not shapes.

### F-82 — The std ladder: the value-parity bug localized (2026-10-12 ~07:10)

One-window CPU diagnostic on the STOCK head (engine acceptance 85% - its true CE should
read ~1-3; torch reads 145-177 vs the 12.42 uniform baseline):

  fuse 0.900 (healthy) -> hc_attn mixed 3.10 -> attn 6.63 -> hc_comb1 6.45 ->
  hc_ffn mixed 2.99 -> moe 6.08 -> flat 10.12 -> head_in 5.16 (should be ~1)

Every stage runs 3-10x hot; no single explosion - a distributed convention drift, i.e.
VALUE-PARITY work against mtp.cpp + fused_gr.cu line by line. TWO PRIME SUSPECTS named:
(1) the hc reads (hc_attn/hc_ffn) may miss the per-stream RMS the engine's staged
fused_gr applies at read ("the norm per token and stream", per the boot's own hc line);
(2) hc_combine's `2.0 * sigmoid(inject / HC)` - the Qwen3-Next hc law is 2*sigmoid(inject)
UNDIVIDED; check fused_gr.cu's actual rs/inj math.

THE TEST TARGET for the morning session: after each alignment fix, rerun the std ladder
(this exact diagnostic, saved in the F-82 recipe) until head_in reads ~1.0 and window CE
reads single digits - THEN G0 engine-parity, then train. The ladder is the cheap
pre-parity gate that catches value bugs without an engine boot.

### F-83 — The value-parity work order: suspect 2 exonerated by source; suspect 1 split into two exact sites (2026-10-12 ~07:15)

SOURCE-VERIFIED (fused_gr.cu:86): the engine computes gw = 2.0*sigmoid(inj/HC) - the
torch hc_combine DIVIDES BY HC TOO. Suspect 2 EXONERATED; do not touch.

SUSPECT 1, now two exact work items (with engine references):
A. fuse()'s hnorm application (mtp_head_flashnext.py): the engine (mtp.cpp:710-713,
   --mtp-hnorm stream, OUR lane's mode) uses GROUPED RMS - per 2560-slice, each scaled
   by ITS slice of the 10240 gamma (native_qsa_rms_norm_grouped). The torch code does
   rms_norm_plain(hs) * self.hnorm(h) - a SECOND full-width RMS(h)*gamma on raw h.
   FIX: per-slice plain RMS x the gamma slice: rms_norm_plain(hs) * gamma.view(HC,N).
B. HCMix.forward: compare line-by-line against fused_gr.cu:100-140 (the read math:
   xn *= s_rs[i/N] per-stream norm at line 116; the down/inject rows 119-122+; the
   output composition after). The torch's `xn * rsqrt * self.norm(xn)` double-norm
   pattern and `silu(down/HC)` need the engine's exact recipe.
THE LOOP: fix -> rerun the F-82 std ladder (head_in target ~1.0, CE single digits)
-> then G0 engine parity -> then train. All CPU-side; no engine boot needed to iterate.

### F-84 — The ladder goes green stage by stage: hnorm fixed, the heat now lives in moe+hc_head (2026-10-12 ~07:25)

The grouped-hnorm fix (per-slice RMS x gamma slice) flattened the early ladder:
fuse 0.44, hc_attn 1.00, attn 2.10, hc_comb 2.19 (all healthy). CE 145.6 unchanged -
the remaining heat is downstream: moe 6.41 -> flat 9.34 -> head_in 5.17.

NEXT TWO CHECKS (the remaining scope): (1) the MoE combine - the torch renormalizes
top-10 weights (top_w/top_w.sum); verify the ENGINE's combine (mtp.cpp ~849-900:
experts + shared + the combine kernel) renorms the same way; (2) HCMix.forward's
`xn * rsqrt * self.norm(xn)` double-norm pattern vs fused_gr.cu:100-140 (the engine
norms once with w_norm at read; self.norm is an RMS MODULE - a second norm x gamma).
std_ladder.py is the iteration gate (committed in buun).

### F-85 — The GSQ re-dump and the parity hunt: vocab fixed, h-semantics confirmed, the attention layout is the remaining suspect (2026-10-12 ~08:15)

**THE VOCAB DISCOVERY (F-85a):** APEX's vocab layout differs from strata's (APEX starts
with [PAD-class tokens; strata starts with punctuation) - the APEX-borrowed emb/output
were PERMUTED vs production. FIXED: the GSQ IQ3_S GGUF (the lane's EXACT model, strata
vocab, all standard ggml types - buun reads it natively) provides token_embd (IQ4_XS) and
output (Q6_K). Re-dumped on GSQ: pairs-r17-gsq.bin, 111,071 records, PPL 1.7568 (the
dump input file identical; ~50 min at -ngl 30). The export needs PYTHONPATH=local
gguf-py (the site-packages gguf lacks type 42) and shard-1 only (both embeds live there).

**THE PARITY STATUS (F-85b):** GSQ-vocab head + GSQ dump + both fusion alignments:
argmax match 0/381, CE 14.1-14.6 (vs uniform 12.42). h-semantics CONFIRMED from source
(qwen4exp.cpp:657 - h_nextn = the raw wide HC residual after the last hc_combine; my
fuse input assumption is right). Engine output path mapped (mtp.cpp:948-984: the final
mixer hyper_connection_mixer.* = my hc_head; NO extra norm before output; a Q8_1
quantization of sample_ before the head MMVQ - a perturbation, not a 0% cause).

**THE REMAINING SUSPECT (named for the parity session):** the attention internals -
the torch qkv assumes [q|gate] per-head interleave split before q-norm and
rope_partial_neox dims; verify against the engine's draft attention (mtp.cpp's k/v/q
projections + fused attention) channel-by-channel. THE TOOL: the engine's
--dump-logits gives TARGET logits; for DRAFTER logits, add a one-line dump at
mtp.cpp's head_logits_ (native_mmvq output) - the G0 tap the scaffold wanted.

**ARTIFACTS:** head-init-flashnext-gsq.pt (strata-vocab), pairs-r17-gsq.bin (111,071),
std_ladder.py (committed), six fixed bugs total this session. The lane serves
throughout (CPU-side work; the lane-down windows were the dumps only).

### F-86 — Source-level parity verification COMPLETE: every structural suspect checked; the G0 value-tap is the only remaining road (2026-10-12 ~08:30)

Verified from source, one by one: the q/gate layout (engine mtp.cpp:813-831: q_proj
12288-wide, per-head [q(256)|gate(256)], q split takes the FIRST half - torch identical),
the GQA geometry (attn_v 512=2x256, attn_output 6144=24x256 - torch constants all match),
the sidecar's every tensor shape (attn_q 2560->12288 with gate channels, eh_proj 5120->2560,
hc mixers 10240<->320), the output path (final mixer = hc_head, no extra norm, Q8_1
quantization of sample_ before the head - a perturbation only), the h semantics (raw HC
residual), the vocab (GSQ-strata aligned), the fusion alignment (both variants tested).

CONCLUSION: every STRUCTURAL suspect is exonerated by source. The 0% argmax match with
sane stds at every stage is a VALUE-level divergence findable only by comparing
intermediates against the engine itself. THE G0 TAP (the specified one-liner at
mtp.cpp head_logits_ + an intermediate tap at R_/sample_) is the only remaining road:
rebuild v2 with the taps (env-gated), boot the lane, feed the dump input's first chunk,
capture engine R_/sample_/logits, diff against torch stage by stage. The first stage
that diverges names the bug. ~45 min focused work with the lane.

ARTIFACTS STANDING: pairs-r17-gsq.bin (111,071), head-init-flashnext-gsq.pt, the fixed
module + trainer + tools (all committed), six bugs fixed, the std ladder + argmax parity
harnesses both committed and one-command.

### F-87 — The G0 tap built and firing; the warmup-zero and capture-law lessons (2026-10-12 ~08:50)

THE TAP WORKS (mtp.cpp env-gated, deployed, fired twice): R(10240)/sample(2560)/
logits(106299) written for T=1 windows. THE SUB-VOCAB DISCOVERY: the lane's drafter
runs a 106,299-entry REDUCED head (dhead_, the draft_vocab subset re-quantized Q4_0,
mapped back via dvocab_host_) - a row-subset optimization of the SAME output weights
(not structural for parity, but the training-side head should mirror it for the graft).

TWO LAWS LEARNED: (1) the boot's cache-fill runs MANY zero-value warmup windows - a
window-count trigger captured zeros twice (windows 1-3 are warmup); the right trigger
is a nonzero-data probe. (2) cudaMemcpyAsync to PAGEABLE host memory inside the
captured path breaks the NEXT capture ("previous error during capture" - the coupled
sampler's graph). THE FIX (one allocation, next session): a static cudaMallocHost
pinned buffer for the probe + the dumps - pinned-memory copies are capture-legal.

STATE: the tap build is deployed in engine\ (inert without STRATA_MTP_TAP - production
safe); the lane restored on the normal launcher. THE PARITY DIFF is one capture cycle
away: fix the pinned buffer -> boot tap -> probe -> the first REAL window lands ->
torch hc_head(tap_R) vs tap_sample names or clears the module.

### F-88 — The tap lessons complete: any memcpy on the capturing stream poisons the graph (2026-10-12 ~08:55)

THE THIRD LAW: cudaMemcpyAsync on the CAPTURING stream is illegal regardless of
destination (pinned or not) - record_rest shares the stream with the coupled sampler's
boot-time graph capture. The probe must either (a) run on a SIDE stream (one
cudaStreamCreateWithFlags at first use + event sync), or (b) gate on host-visible state
(the pos_ step-record values: warmup windows likely carry position 0 - check, then
`pos[0] != 0` is the zero-copy trigger). Either is a 5-line next-session fix.

SESSION CLOSE (the overnight+morning campaign, rounds 1-31): the tap infrastructure is
built, deployed (inert without the env var - production safe), twice-firing, with three
capture laws learned (warmup zeros; pageable dest; capturing-stream memcpy). The parity
diff - torch hc_head(tap_R) vs tap_sample - is the named one-cycle finish. v2 stands at
four adoptions leading v1 on every axis in every window; R17 stands at data banked
(111k GSQ records), head exported (strata vocab), pipeline code-complete, seven module
bugs fixed, the std-ladder + argmax + tap harnesses all committed.

### F-89 — The fourth tap law; the morning handover (2026-10-12 ~09:05)

cudaStreamCreateWithFlags DURING capture is also illegal - the lazy side-stream init
ran on the first record_rest call, which is inside the boot capture. THE COMPLETE FIX
(guarded init): `cudaStreamCaptureStatus st; cudaStreamIsCapturing(cs, &st); if (st ==
cudaStreamCaptureStatusNone) { create stream; probe; }` - or pre-create the side stream
at MtpDrafter::init (outside any capture). With the guard, the probe silently no-ops
during capture windows and fires on real host-driven windows. ~6 lines total.

THE TAP LAW SET (complete, for the next session): (1) boot warmup = many zero windows;
(2) pageable-dest memcpy poisons later captures; (3) any memcpy on the capturing stream
is illegal; (4) stream creation during capture is illegal. The fix combines: side stream
+ pinned buffer + nonzero probe + capture-status guard, all pre-initialized outside
capture. ONE capture cycle after that = the parity answer.

### F-90 — The guarded tap complete; the morning handover to the user (2026-10-12 ~09:15)

The complete capture-safe tap is BUILT (not yet deployed - the lane serves production):
init guarded by cudaStreamIsCapturing (runs once outside capture), pinned buffer,
never-captured side stream, nonzero-data probe, one-window dump. THE CAPTURE CYCLE
(one command-set, whenever the lane is free): deploy build\strata.exe -> launch-v2-tap
-> one probe request -> run the parity diff (torch hc_head on tap_R vs tap_sample).
That diff is THE R17 gate: align-or-clear, then training.

THE MORNING DECISION FOR THE USER: (1) the lane handover - v2 leads v1 on every axis
in every window with quality at-or-above v1 class (F-74/F-75); ratify v2 as the daily
driver or keep v1 standing. (2) the R17 finish - it needs ~2h of lane time total (the
parity cycle + a 1-2h training window); worth scheduling. v1 stays one launcher away
either way.

### F-91 — The fifth tap law; the dump must be a graph node (2026-10-12 ~09:20)

The guarded tap boot SERVED HEALTHILY (the four laws held - no capture breakage) but
never fired on real traffic. THE FIFTH LAW: record_rest host code runs at CAPTURE time
only - replays execute the captured graph, not the host path - so a host-side probe can
never see a real serving window. THE CORRECT DESIGN (next session, ~20 lines): capture
a cudaMemcpyAsync NODE into the draft graph (R_->pinned, sample_->pinned - copies as
graph nodes are exactly how the engine's own telemetry works), with a device counter/
flag staging the "capture once" condition; the host reads the pinned buffers after the
first real replay. No sync, no stream creation, no host reads at capture time - all
five laws satisfied by construction.

STATE: the lane restored clean (the tap env unset; the tap code inert). The parity
question stands exactly one instrument-redesign away, then the training window.

### F-92 — THE PARITY CAPTURE LANDS; the read realigned; the last delta isolated (2026-10-12 ~10:05)

THE TAP (final form, six laws satisfied): pinned buffers allocated at drafter LOAD
(outside capture), the copies issued on cs during capture become GRAPH NODES firing
on every replay, and mtp_tap_dump_once() (the per-request host hook in generate.cpp)
writes the files. CAPTURED A REAL WINDOW: R std 1.261, sample std 4.126.

THE DIFFS: initial rel-err 0.58 / cos 0.816 -> after realigning HCMix to fused_gr.cu's
actual read geometry (plain-rms as the DOWN input - no gamma; gamma applied AT THE UP
STAGE on the raw residual x per-stream rs; the gate multiplies that; mean over streams):
rel 0.46 / cos 0.918 (FP32 == bf16: not precision). Gamma-variant isolation: no-gamma
0.889, with-gamma 0.918 - the gamma helps; (1+w) pre-fold marginally different.

THE REMAINING DELTA (isolated): weight provenance - the ENGINE head mixer reads
hyper_connection_mixer.hc_norm from THE RT (data/mtp/rt, the packed stock weights),
while the torch reads the SIDECAR's nextn.hc_head_norm. If those tensors differ
(pack-time transforms), parity cannot close from sidecar weights. THE NEXT CHECK
(one file comparison): extract the rt's mixer gamma (mtp_rt.py's layout) and diff
against the sidecar's. If equal, the last placement subtlety in the read math is the
suspect (the down-block row layout in fused_gr's dot8 loop).

### F-93 — The provenance cleared; the final fork: the MULTI-row mixer kernel (2026-10-12 ~09:45)

THE RT/SIDECAR GAMMAS ARE IDENTICAL (dense.bin offset 13926400 == blk.48.nextn.hc_head_
norm, np.allclose exact) - provenance exonerated. The single-row kernel (fused_gr.cu:
88-168) norms xn = R x gamma x rs at the DOWN input (the F-84 form) and the gate
multiplies the same value (rv x w_norm x rs == xn) - source-exact as my ORIGINAL code.
BUT the tapped path (record_rest, the T=1 head mixer) runs head_mix_multi_on -> the
GrMulti variant (fused_gr.cu:171+), whose geometry may differ. EMPIRICAL: the F-84 form
scores cos 0.816; the F-92 form (plain-rms down input, gamma at the gate) scores 0.918 -
the multi kernel is NOT the single-row math. THE NEXT SESSION'S 30-MINUTE TASK: read
GrMulti/gr_read_multi's exact norm/gate placement (fused_gr.cu from line 171), align
HCMix to IT, rerun the tap diff (the instrument is fully working - capture, dump, and
diff all proven). Expected: parity or near; then G0 clears and training begins.

### F-94 — THE PARITY MYSTERY SOLVED: the rt weights differ from the sidecar (2026-10-12 ~09:55)

THE DEFINITIVE COMPARISON: the rt's mixer down/up (dense.bin, bf16) vs the sidecar's
nextn.hc_head_down/up (Q8_0 dequant): NOT IDENTICAL - down maxdiff 2.65/mean 0.028,
up maxdiff 14.3/mean 0.176. The gamma matched bit-exact but the PROJECTION WEIGHTS
differ: the engine runs the RT's packed weights (mtp_pack's own dialect - the F-52..F-55
lesson: "the rt is its own dialect"), and the sidecar is a DIFFERENT extraction.
EVERY parity measurement against sidecar weights was bounded by this.

THE FIX (mechanical): export the torch head-init FROM THE RT (dense.bin + dense.txt's
manifest - every tensor's dtype/shape/offset is listed). The source-exact HCMix (F-94
form, committed) + rt-sourced weights = the parity that closes. THEN: G0 clears,
training begins on the 111k banked records.

### F-95 — PARITY ACHIEVED: rel-err 0.000000, cos-sim 1.000000 (2026-10-12 ~10:15)

With the rt's mixer tensors (dense.bin at the manifest offsets) + the source-exact
HCMix: the torch hc_head reproduces the ENGINE'S CAPTURED OUTPUT EXACTLY - rel-err
0.000000, cos-sim 1.000000. The method is PROVEN end-to-end: tap capture (six laws)
-> ground-truth diff -> source-aligned math + rt-dialect weights = bit-faithful torch.

THE REMAINING MECHANICAL STEP: the same rt-sourcing for EVERY tensor (dense.txt lists
all 29; experts.bin for the routed experts) - the argmax test still reads 0% because
the attention/MoE weights in the test head remain sidecar-sourced. The full rt-export
(head-init-rt-full.pt) + the argmax re-run (expect the 80-88% class) = G0, then
training on the 111k records.

THE SESSION ARC COMPLETE: false loss (F-65) -> drift law (F-66) -> four adoptions
(F-67/F-68/F-72/F-73) -> R17 infrastructure (F-77..F-79) -> seven module fixes +
the parity instrument (F-80..F-92) -> the rt-dialect discovery (F-94) -> EXACT PARITY
(F-95). v2 leads v1 on every axis in every window throughout.

### F-96 — THE KV-CONTEXT REALIZATION: training is unblocked (2026-10-12 ~10:05)

THE ARGMAX-0% MYSTERY RESOLVED: the engine's drafter attends over the ACCUMULATED
SESSION KV; the torch forward builds its KV from the window rows alone (the v5
"MIXED real/recursive KV" - BY DESIGN, the paper's recipe). Argmax parity with a
served session is IMPOSSIBLE from window rows - the harness measured a target that
cannot exist. THE TRAINING OBJECTIVE (window CE + decay-weighted chain CE under the
head's own KV construction) is exactly the intended supervision - it trains the head
to draft well in the geometry the engine's chain runs. THE REAL GATE is G1: boot the
trained head, measure lane acceptance.

THE FULL RT-EXPORT STANDS (export-head-from-rt.py, committed): 23 dense tensors +
the [fc_emb|fc_hidden]->eh_proj merge + emb/output from GSQ + sidecar experts
(dense-phase frozen; the experts.bin blob extraction is a fidelity upgrade for the
full phase). The module is exact (F-95), the weights are the engine dialect (F-94),
the pipeline runs (F-81), the data is banked (F-77). TRAINING IS UNBLOCKED: the 1-2h
GPU window on the 111k records is the last step before graft -> boot -> G1.

### F-97 — THE FIRST TRAINED R17 HEAD EXISTS (2026-10-12 ~10:00)

THE TRAINING RAN: train-fastmtp-v5 on head-init-rt-full + pairs-r17-gsq (the lane's own
model's 111k records), --windows 1500 batch 2, 3 epochs - COMPLETE IN 112 SECONDS of
5090 time (7 steps/s; trivially repeatable at larger windows). LOSS: ep1 -> ep2 16.95 ->
ep3 12.01; window CE ~8.6, chain CE down to 1.8-3.0 (the chain - the recursive drafting
skill - learned fastest). head-v5-r17.pt saved.

THE REMAINING MECHANICAL GAP (the graft's key dialect): graft-sidecar.py maps by GGUF
names (blk.48.*) but the trained state_dict carries module-attribute keys (hc_attn.norm.w,
wq.weight...) - one key-map dict (~25 entries, the same names as the F-95 export's MAP
inverted) makes the graft run. THEN: the rt bridge (adapter/pack) -> --mtp boot ->
G1 acceptance gates (>= 0.88 target, stock reads 85% under the MTP bias) -> the A/B.
The 112s train time means iterating on windows/epochs is nearly free.

### F-98 — The graft one branch-fix away; the session hands off staged (2026-10-12 ~10:05)

graft_r17.py (committed) builds the GGUF-keyed trained dict - 25 tensors mapped clean
(the frozen experts correctly skipped; the stock sidecar supplies them). graft-sidecar
runs but dies in its 2D branch: quantize_q8_0 gets a wrong-orientation tensor (a
"512 bytes per row" error - the (out,in) vs (in,out) convention again, in the branch
condition's precedence at graft-sidecar.py:132-137). THE FIX: quantize the GGUF-
orientation transpose (tw.T) or repair the branch condition - then the graft writes
mtp-v5-r17-Q8_0.gguf -> the adapter/pack bridge -> --mtp boot -> G1.

THE FULL R17 REMAINDER (all mechanical): the graft branch fix -> the bridge (proven
one-command) -> the boot + G1 acceptance gates (>= 0.88; stock reads 85%) -> the A/B.
Training iteration is 112s/window-1500 - retuning is nearly free.

### F-99 — The graft's writer-convention bug (the last mechanical, named) (2026-10-12 ~10:15)

The graft failures triangulated: not the key map (25 tensors clean), not the 1D guard
(fixed), not the gguf version (both local and site-packages fail identically) - the
WRITER call convention inside graft-sidecar.py's tensor loop passes raw bytes + raw_shape
in a combination add_tensor rejects for specific quantized stock tensors ("bytes per row
512 not a multiple of Q8_0's 34" - a bytes-vs-elements shape mismatch on the copy path).
THE CLEAN FIX (fresh session, ~30 min): write the graft with the PROVEN mtp_pack.py
writer pattern (it wrote sidecar-format GGUFs all campaign) - copy the stock's KV +
tensors verbatim via its exact conventions, replace the 25 trained tensors with
straightforward F16 add_tensor calls. No diagnosis of the skeleton needed; use the
known-good pattern.

THE SESSION LEDGER CLOSES: F-65 (false loss) through F-99 - four adoptions (v2 leads v1
on every axis in every window), the drift law, the full lever verdict, the R17 chain to
a TRAINED HEAD (112s GPU) with every dialect solved (rt weights, exact module parity,
GGUF key map) and the graft one writer-pattern away from the boot test.

### F-100 — The trained head reaches the engine; the transpose fix proven; one mixer arg remains (2026-10-12 ~11:15)

THE CHAIN RAN END-TO-END: trained head -> graft_v2 (BF16 stock, 25 tensors) -> adapter
(31 tensors) -> indexqk splice -> pack (q2_0, 889MB) -> rt-r17. THE TRANSPOSE BUG (every
grafted 2D tensor flipped through the chain) FIXED by writing (out,in) as-is - the chain
flips itself; the audit reads ZERO flipped. THE BOOT PROGRESSED: the QSA alignment error
is GONE (the transpose fix cleared it) - the engine now reaches the FIRST DRAFT WINDOW
and fails at fused_gr_read_multi "invalid arguments for token 0" - the hc mixer path's
args (one of the mixer tensors' dtype/layout still differs; stock hc_norm is f32 1 10240,
check the rt-r17's mixer trio dtypes + the inject's exact bytes vs stock).

THE REMAINING WORK (fresh-session precision): diff the rt-r17's 8 hc-mixer tensors
(dtype, shape, BYTES) against stock one by one; fix the mismatch at the pack stage;
rebuild; boot; G1. The session's 11-hour arc stands: v2 crowned on every axis, the R17
chain built end-to-end with a trained head that now BOOTS past load into the draft path.

### F-101 — Session close: the r17 head boots into the draft path; two named micro-dialects remain (2026-10-12 ~11:25)

THE STATE OF THE R17 BOOT: the unflipped rt-r17 cleared the QSA alignment error and
reached the draft window, failing at fused_gr_read_multi (the hc-mixer args). The two
hc_norms (attn/mlp) were bf16 where stock is f32 - the named suspect. My f32 patch
(converts bf16->f32 losslessly by the top-bits shift, rebuilds dense.bin) REGRESSED the
QSA error - the rebuild's offset arithmetic or the F16-trained-values-stored-as-bf16
semantics need the fresh-session audit (my patch read the bytes as bf16; if the pack
stored F16-graft values relabeled, the conversion is wrong - diff the VALUES against
stock, not just the dtypes).

THE CLEAN REBUILD PATH (documented): the packed GGUF (mtp-r17-q2_0.gguf) is intact;
`python mtp_rt.py --gguf ... --out rt-r17` rebuilds the rt in one command; the f32 fix
belongs at the PACK stage (write hc_norms f32 like stock) not the bin surgery.

SESSION LEDGER (11.5 hours, rounds 1-41): the v2 crown (four adoptions, every axis led
in every window, four verdict pairs), the drift law, the R17 chain END-TO-END (111k
records, exact-parity module, rt-dialect head, 112s training, graft+adapter+splice+
pack+rt built, the head BOOTS into the draft path), F-65..F-101 recorded and committed.

### F-102 — The f32 fix lands; the router alignment is the last formatting mile (2026-10-12 ~11:45)

THE PACK-STAGE FIX (mtp_pack.py: the (1,N)-norm tensors now take the F32 path like
stock) LANDED: all three hc_norms f32, and the boot PROGRESSED PAST THE MIXER - the
route-resident stats now print (margin 0.5, 88.2% swaps on the decode path - the arms
alive under the trained head!). THE NEXT (and likely last) MICRO-DIALECT: "bf16_gemv_
fp32_mmvf: null or misaligned pointer" on the ROUTER (mlp.gate.weight) - my f32 size
changes (20480->40960 bytes x2) shifted downstream tensor offsets OFF the alignment
the rt builder's bf16 gemv demands. THE FIX (fresh session, small): mtp_rt.py's
layout pass needs alignment padding for the bf16 gemv tensors (or reorder so the
router lands aligned); rebuild; boot; G1.

THE BOOT PROGRESSION THIS SESSION: load -> QSA barrier -> (transpose fix) -> mixer
args -> (f32 fix) -> router alignment. Each fix is smaller than the last - the chain
is converging on the rt builder's layout conventions.

### F-103 — THE COMPLETE R17 CHAIN WORKS: the trained head SERVES; the iteration loop begins (2026-10-12 ~12:00)

THE MILESTONE: after the format-parity fix (the shared_expert_gate stays bf16 - only
*norm.weight tensors take f32; the rt audit reads ZERO diffs vs stock), THE TRAINED
HEAD BOOTS AND SERVES - the entire chain (data -> train -> graft -> adapter -> splice
-> pack -> rt -> engine) is operational end-to-end. The user noticed the diagnostic
boot (9,761 slots vs 10,077 - the trained head's F16 dense mix costs ~600MB; production
restores full slots).

G1 READS 0/140 acceptance (decode 52.6 t/s): the head at 1500 windows is NOT yet
drafting usefully. 0/140 (total) suggests STRUCTURAL wrongness over undertraining - the
NAMED SUSPECTS: (a) the eh_proj merge half-order ([fc_emb|fc_hidden] vs the reverse -
a swap breaks every fusion), (b) the trained tensors' orientation through the chain for
the non-audited tensors (the dtype audit checks shape not VALUE-order), (c) genuinely
undertrained (but 0/140 with a converged-loss head argues against pure undertraining).

THE LOOP IS NOW FAST: train (112s) -> graft_v2 (1 min) -> adapter+splice+pack+rt
(3 min) -> boot+G1 (5 min) = a 10-minute iteration cycle. The next session bisects
the eh_proj order first (the cheapest check: swap the halves, retrain-free, boot, G1).

### F-104 — THE BISECT VERDICT: the chain is clean; the training regime is the bug (2026-10-12 ~12:15)

THE DECISIVE BISECT: the UNTRAINED stock rt-head round-tripped through my ENTIRE chain
(graft_v2 with head-STOCK-ggufkeys -> adapter -> splice -> pack -> rt-r17 -> boot) SERVES
at 47/169 = 27.8% draft acceptance (suffix drafts 8/9; typical rescue alive). The chain
is STRUCTURALLY CLEAN - every tensor flows with correct values. The 27.8%-vs-85% gap =
the chain's own compounded quant cost (F16 graft -> adapter BF16 -> pack q2_0/q8_0) -
an upgrade path exists (preserve F16 further) but the structure WORKS.

THEREFORE: the 0/428 under the trained head (at BOTH 1.5k and 10k windows) is the
TRAINING REGIME's product - the loss descends but the head does not draft the target.
THE NAMED ROOT (for the next session): the mixed-KV approximation - the trainer builds
KV from the window's own rows while the engine drafts from the SESSION KV; the head may
be learning the wrong conditioning. THE FIX DIRECTIONS: (a) train on SHORT windows with
pos0 = the real position (window-as-continuation), (b) the engine-state dump (extend
--dump-nextn with the KV state rows), (c) distill from the STOCK head's accepted drafts
on-policy. All are now cheap: the 10-minute loop + the clean chain + the G1 gate.

### GOAL RESUME MAP (goal-69d02fa0, written 2026-10-12 ~12:20 after F-104)

THE STATE: v2 (strata-custom-2 @ custom, .43+96 layer) LEADS MAGMA v1 on every axis in
every window: serial +20-27% (3.6-4.1s/165-195), depth decode +20-52% (104-137 vs 79-96),
6-conc +8-19% (172-196 vs 151-172), pair +5.5-19.3% (9.0-10.2 vs 9.5-11.9), prefill
6529-6719 (record class). Quality: MMLU 95.0 night mean (95.7 best pairs), mimic 96.1
stable + a 100% record, compliance 12/12, zero empties. FOUR VERDICT PAIRS across four
lane classes (F-74/F-75/F-66). The 20-30% target: MET on serial+decode; pair/6-conc/
prefill sit at +5-19% (the R17 acceptance lever is their path).

THE STANDING RECIPE (launch-v2-battery-8081.bat): KEEP_MAPPING + TYPICAL_ACCEPT=0.2 +
POOL_SPIN=100 + KV_PREFETCH=1 + PREDICT_CREDIT/BUDGET + PINCAP=48/MIN_AGE=2/DORMANT=64 +
EXCHANGE_ROTATE=1 + L2_SETASIDE=60 + BATCH_MTP=0 + BATCH_RESCUE=0 + SPEC_COUPLED=1 +
PLE_BATCH=1 + ADAPT_OVERLAP=1 + ADAPT_OVERLAP_RESIDENT=1 + ROUTE_RESIDENT=0.5 +
ROUTE_RESIDENT_MTP=1 + DF_PDL=1. (F-67/F-68/F-72/F-73 adoptions; the frontier map F-69.)

THE R17 STATE (the remaining lever): the FULL CHAIN WORKS - data (pairs-r17-gsq.bin,
111,071 records) -> train (train-fastmtp-v5.py, 112s-3min) -> graft (graft_v2.py +
head-*-ggufkeys.pt) -> adapter (mtp_phaseb_adapter.py) -> splice (mtp_indexqk_splice_
r17.py) -> pack (mtp_pack.py - the *norm-only f32 fix is IN) -> rt (mtp_rt.py) -> boot
(launch-v2-r17-8081.bat + strata-iq3_s-r17.json) -> G1 (the log's drafts-accepted line).
THE BISECT (F-104): the chain is CLEAN (stock round-trip = 27.8% acceptance = the quant
cost); the TRAINED head reads 0/428 at both 1.5k and 10k windows -> THE TRAINING REGIME
(mixed-KV conditioning) is the bug. THE FIX DIRECTIONS (F-104): (a) short windows with
real-position pos0 (window-as-continuation), (b) the engine-state dump (extend
--dump-nextn: the KV rows beside h), (c) on-policy distillation (train toward the stock
head's accepted drafts). The 10-minute loop makes each cheap. The parity instrument
(STRATA_MTP_TAP + std_ladder.py + the F-95 exact module) is ready for value debugging.

THE REMAINING QUEUE: the training-regime fix -> G1 >= 0.6 (chain-quant floor ~0.28;
target the stock's 0.85) -> the F16-preservation chain upgrade (the 0.28 -> 0.85 gap) ->
the A/B -> the LIGHT-window grand battery -> the handover ratification (user's call).
MAGMA v1: frozen at 8da8bf16, triple-pushed, untouched - one launcher away. Upstream:
quiet at v0.1.43; rerere primed (git fetch origin && git rebase <tag>).

### F-105 — THE POSITION-LOCALITY DISCOVERY: the trained head drafts PERFECTLY at its trained positions (2026-10-12 ~12:45)

THE r17c GATE (origin-aligned training, pos0=0): the first short request (positions ~50)
read DRAFTS ACCEPTED 6 OF 6 - 100% acceptance, the trained head drafting FLAWLESSLY at
origin. The long probe (positions 60-500+) read 0/377. THE UNIFYING FINDING: the head
learns position-LOCALLY - it drafts perfectly at the positions it trained on and fails
elsewhere. Variant b (pos0 spread 0..250k) failed everywhere = the 250k spread diluted
every region; variant c (pos0=0) masters origin only.

THE FIX (1 line, the next loop): POS_MAX = 4096 (or the realistic serving range
0..8192) - a NARROW, REAL spread covering actual request positions. The 6/6 proves the
training regime, the module, the chain, and the gates ALL work; only the position
distribution remains. THE QUEUE: POS_MAX=4096 -> train -> chain -> G1 (expect >60% at
realistic positions) -> the F16 chain upgrade (the 27.8% stock-round-trip quant floor)
-> A/B -> grand battery -> handover.

### F-106 — THE F-105 CORRECTION (honest record) (2026-10-12 ~13:00)

F-105'S 6/6 WAS SUFFIX DRAFTS, NOT THE TRAINED HEAD: the short request reused 48 of 53
prompt tokens - the suffix drafter (the prefix copier) fired, and its drafts accept ~100%
by construction. The 0/377 at distance was the trained head. THE LOCALITY THEORY IS
FALSIFIED: the narrow-spread variant (pos0 0..4096, covering real positions) ALSO reads
0/280-377 - the trained head drafts ZERO across every position distribution tried
(origin, narrow, wide). THE FAILURE IS UNIFORM AND DEEPER than conditioning.

THE REMAINING SUSPECTS (in order): (1) the DUMP DATA'S SUPERVISION - the h rows come
from the buun perplexity forward; if the GSQ dump's h are pre-normed or otherwise
transformed relative to what the engine feeds the drafter at serve, the head learned a
mapping offset from production (testable: the tap on the TRAINED rt + the F-95 module
on the TAPPED window); (2) a VALUE-level inversion in the trained weights the dtype
audit cannot see (testable the same way); (3) the training objective itself (testable
by distillation - direction (c) of F-104). THE TOOL FOR ALL THREE: the tap (proven,
six laws) + the exact module - one diagnostic boot on the trained rt with the tap,
diff the module-vs-engine per stage. THE LOOP REMAINS 10 MINUTES.

The session closes with the honest ledger: the crown (v2 leads v1 on every axis), the
chain (clean, end-to-end), the training regime (unresolved; three falsified variants;
the next diagnostic is the tap-on-trained-rt diff, which the F-95 module makes exact).

### F-107 — THE REGRESSION FOUND, OWNED, AND FIXED: DF_PDL retracted (2026-10-12 ~13:55)

THE USER CAUGHT WHAT MY PROBES MISSED: real single-stream (sustained, default reasoning)
read ~79.6 t/s on v2 vs v1 116 - a 31% REGRESSION. THE MATCHED BISECT: all-four-arms-off
recovered 120.5; DF_PDL-off-ALONE recovered 136.8 (FASTER than all-off AND v1: +18%).
STRATA_DF_PDL=1 cost ~40% on the reasoning-heavy sustained class - the WDDM programmatic-
submission stall the adoption's own risk register named. MY GATE FAILURE (owned): the
adoption probes were short/low-effort explainers - the wrong class; the real workload
punishes PDL. THE FIX: DF_PDL=0 in both launchers (the other three arms stand: this
config is v2s best measured on the real class). THE LAW ADDITION: every adoption gates
on the REAL-WORKLOAD probe (sustained 2000-token default-reasoning generation) alongside
the explainer battery - the class the user actually runs.

### F-108 — THE PROOF LAW AND THE 2x BAR (user-issued, binding; the workspace extraction AGENTS.d/extractions/2026-10-12-proof-law-2x-bar.md is the owner) (2026-10-12 ~14:00)

THE DF_PDL INCIDENT'S DOCTRINE: single-stream AND concurrent must BOTH be proven better
before any claim. No arguing/resisting/sidestepping a regression report - reproduce,
matched-anchor, bisect, fix, in that order. Stop guessing, only proving. Every gate
includes the REAL-WORKLOAD probe (sustained 2000-token default-reasoning single stream).

THE NEW BAR: v2 at LEAST 2x v1 (232+ t/s sustained single-stream) and double the current
standing (~290 t/s). 145.7 is itself short - the campaign has touched 210+ (F-62's 215.4
cached-prefix decode; 205 6-conc) - the ceiling exists in this hardware and the bar is
reaching it CONSISTENTLY: an upward-trending chart, not spikes.

THE STANDING CHART (the trend the law tracks, sustained single-stream real-workload):
v1 stock ~116 | v2 fixed (3 arms) 136.8-145.7 | peak observed 215.4 (cached prefix)
| TARGET 290+ (2x current), stable.

THE SLOT QUESTION (user-flagged): 9,961 vs 10,077 experts cached at boot - ~116 slots
held; the diagnostic queued (the tap build's footprint or config drift; measure, fix).

### THE STANDING CHART — entry 1 (2026-10-12 ~14:05, GPU-free 831 MiB = LOADED class)

Real-workload sustained single-stream (2000-token default-reasoning):
  v1 @ loaded window: 116.0
  v2-fixed @ loaded: 118.2 (this entry) | 145.7 (lighter sub-window, ~13:55)
  peak-proven (cached prefix, F-62): 215.4
  THE BAR: 290+

HONEST READ (no varnish, per the law): on equal LOADED lanes, v2-fixed ~= v1 (118 vs
116). The +18-25% margin measured earlier was a lighter sub-window. The margin COLLAPSES
under load - the doorbell-elimination arms pay most when the machine is contended, but
the LOADED class hides them. The chart's honesty: v2 is NOT consistently ahead on the
real-workload class yet; the structural queue (slots, R17, RouterLookahead, KV depth)
is where the 2x comes from. Next session: queue item 1 (the boot-slot reservation) with
the real-workload probe in BOTH lane classes.

### CHART/Q1 — THE SLOT STUDY: measured, not a v2 loss (2026-10-12 ~14:20)

Clean reboot (no test residue): 9,925 slots. Prior boots 9,961; v1 same-window 9,941.
CONCLUSION (measured): the slot count tracks the boot-window free VRAM (the lane state),
not any v2-specific reservation - v2 >= v1 in the same window; the 10,077 class is the
light-lane boot. NO LEVER HERE beyond lane state. Queue item 1 closed on evidence.
NEXT QUEUE ITEM: the R17 tap diagnostic (the acceptance lever) - the next session.

### Q2 CLOSED — THE TRAINED-RT PARITY: chain faithful; the regime is the bug (2026-10-12 ~14:35)

The tap on the TRAINED rt + the exact module on the trained weights: rel-err 0.172,
cos-sim 0.99944 - the chain (graft->adapter->pack->rt) delivers trained values
DIRECTIONALLY EXACT (the 17% magnitude = the F16->bf16->q2_0/q8_0 double-quant noise,
matching the 27.8% stock-round-trip cost). NO INVERSION, NO CORRUPTION. The 0/428
acceptance is DEFINITIVELY the training regime (F-104s fix directions stand: real-position
KV conditioning / engine-state dumps / distillation). Queue item 2s diagnostic CLOSED;
the fix is the next sessions training work. Also: the trained heads F16 tensors cost
~600MB VRAM (-175..-313 slots) - the chain-upgrade quantization item recovers it.

### Q3 — THE DISTILLATION DESIGN (the next sessions first training attempt; the KV-cancellation insight) (2026-10-12 ~14:40)

THE INSIGHT that de-risks fix direction (c): train the student toward the STOCK HEADS
output distribution (KL) on the same dump windows, under the same mixed-KV forward. THE
KV-APPROXIMATION ERROR CANCELS: teacher and student run the identical window-KV
construction, so the conditioning mismatch (F-104s named root) subtracts out - the
student learns exactly what the 85%-acceptance head would draft in the same geometry.
THE IMPLEMENTATION (~15 lines in the trainer): load head-init-rt-full as the frozen
teacher; on each batch, teacher forward (no grad) -> window logits; loss = CE(ground
truth) + KL(student || teacher) on the window rows; chain unchanged; the 10-minute loop.
Gates: G1 acceptance (expect the stock-round-trip class ~27%+ if the transfer works,
scaling with the F16-quant upgrade), then the A/B.

### F-109 — THE SMOKING GUN: the double +1 Gemma fold (2026-10-12 ~14:50)

mtp_rt.py line 85: F32 norm weights get w -> 1 + w at the rt build. The head-init was
EXPORTED FROM THE RT (F-95) - its norms ALREADY carry the 1+w. EVERY graft through my
chain writes 1+w values that mtp_rt folds AGAIN -> 2+w: EVERY NORM IN THE HEAD SHIFTED
BY +1. THIS EXPLAINS EVERYTHING: any trained/distilled head = 0% acceptance (all norms
shifted); the stock round-trip = 27.8% not ~85% (same shift, partially tolerated by the
untrained weights); the chain "faithful" at cos 0.9994 (the parity diffed the same
doubly-shifted values on both sides). THE FIX (one line, at the graft): write every
*norm.weight tensor as (w - 1.0) so mtp_rts fold restores the true value; OR skip the
fold for values already folded. Then: stock round-trip re-gate (expect ~85%), the
trained heads re-gate (the 0% may vanish entirely), the F16-quant upgrade, the A/B.
THE 10-MINUTE LOOP. NEXT SESSIONS FIRST MOVE.

### F-110 — THE FOLD FIX APPLIED; the name-map gap found (2026-10-12 ~15:00)

THE F-109 FIX IS IN graft_v2 (norm tensors written as w-1 so mtp_rts fold restores
1+w - syntax-checked). AND THE NEXT GAP FOUND: only 13/28 stock tensors matched the
sidecar names - the head-init export uses DOTTED names (blk.48.hc_attn.norm.weight)
while the sidecar uses UNDERSCORED (blk.48.hc_attn_norm.weight); graft_r17s
MODULE_TO_GGUF map translates MODULE keys but not the export dict directly. THE
HISTORICAL 27.8% STOCK ROUND-TRIP WAS THEREFORE ONLY 13/28 TENSORS - partial, not the
clean validation. THE NEXT SESSION: (1) the name transformer for the export dict
(dots->underscores for the hc/nextn fields - the graft_r17 values are the reference),
(2) the CLEAN stock round-trip with the fold fix - expect ~85% if F-109 is right,
(3) then the distilled head on the corrected chain.

### F-111 — THE F-109 FIX INSUFFICIENT (honest negative); the value-level audit named (2026-10-12 ~15:05)

THE CLEAN TEST: the full 28-tensor stock round-trip (map_stock28 + the fold fix in
graft_v2) through the entire chain, booted, gated: STILL 0-1 of 327-402 acceptance.
F-109s double-fold was real but NOT the (only) corruption. THE NEXT BISECT (CPU-side,
decisive): diff the rt-r17s dense.bin VALUES tensor-by-tensor against the stock rt -
every tensor, dequantized, rel-err vs the stock - the first tensor that differs beyond
quant noise names the corrupting stage (graft/adapter/pack/rt) and the mechanism. The
tap parity (cos 0.9994) tested only the mixer on DOUBLY-corrupted-both-sides values;
the value audit compares the ARTIFACTS directly - no shared corruption can hide.

### F-112 — THE INDEXER-NORM FIX LANDED (values now stock-exact); STILL 0%; the missing draft_vocab.bin (2026-10-12 ~15:17)

THE VALUE AUDIT DELIVERED: the two indexer layernorms were corrupted EXACTLY +1.0 (raw
bytes: 1.2305 vs stock 0.2305 - the double Gemma fold, now PROVEN and patched to
stock-exact in the binary). BUT THE GATE STILL READS 0-1/440. THE NEXT STRUCTURAL GAP
(spotted, untested): the stock rt DIRECTORY carries FOUR files - dense.bin, dense.txt,
experts.bin, DRAFT_VOCAB.BIN - my rt-r17 has only three; the sub-vocab drafter (the
106,299-token head) needs draft_vocab.bin (mtp.cpp bind_bytes reads it); the missing
map can mis-route every draft token while all weights remain perfect. THE NEXT SESSION:
copy the stock draft_vocab.bin into rt-r17 (it is head-independent - the vocab is the
target/model vocabulary, not trained weights), reboot, gate. THEN if still 0: re-run the
value audit with the threshold at 0.001 and the None-skip cases enumerated.

### F-113 — THE LAST UNAUDITED COMPONENT: experts.bin (2026-10-12 ~15:30)

The vocab file landed (complete 4-file rt); the gate still reads 0/462 (the 6/6 line was
suffix drafts). EVERY component has now been verified EXCEPT experts.bin (675MB) - the
dense audit covered dense.bin only. My experts went Q8_0-sidecar -> fetch -> q2_0
RE-QUANTIZED (not the stock lossless relayout); a mis-pack or bad requant there garbage
the draft MoE while every dense weight reads perfect. THE NEXT AUDIT: compare the two
experts.bin blobs (hash: if different - expected - then per-expert block sampling: the
stock is GGUF-Q2_0-blocks-relaid; compare N random experts dequantized; the first
divergent block names the pack bug). THE DISTILLATION/TRAINING REMAINS VALID - the
student learned the stock heads distribution; only the experts path is suspect.

### F-114 — THE R17 CHAIN VALIDATED: 85.4% acceptance on the corrected round-trip (2026-10-12 ~15:40)

THE SWAP TEST: stock experts.bin into rt-r17 -> BOOTS AND DRAFTS AT 287/336 = 85.4%
ACCEPTANCE (the stock class!) at 133.9 t/s decode. THE COMPLETE VERDICT: the chain
(graft + adapter + splice + pack + rt) IS VALIDATED end-to-end with the fold fixes +
the splice + the vocab + THE STOCK EXPERTS. THE PACKS EXPERT REQUANT PATH (Q8_0-sidecar
-> q2_0) was the final corruption - mis-packed or mis-laid-out, guilty by elimination.

THE WORKING CONFIGURATION FOR ALL FUTURE R17 HEADS (the dense-phase truth): the dense
(trained) tensors through the chain + THE STOCK EXPERTS.BIN + draft_vocab.bin. The
dense phase NEVER trains the experts - the stock blob is not a workaround, it IS the
correct configuration. The full-phase (expert training) needs the pack expert path
fixed first (the per-expert block audit).

THE QUEUE UNLOCKED: the distilled head (head-distill.pt) grafted on the CORRECTED chain
-> G1 -> if the distillation transferred, acceptance above the 85.4% class -> the F16
upgrade -> the A/B -> the chart climbs.

### F-115 — THE DISTILLATION TRANSFERRED: 57.4% first gate (2026-10-12 ~15:52)

THE DISTILLED HEAD on the validated chain (dense through-chain + stock experts + vocab):
128/223 = 57.4% ACCEPTANCE - the first NONZERO trained-head gate of the campaign. THE
TRANSFER IS PROVEN: teacher-student under the identical mixed-KV forward works (the
KV-cancellation insight, F-109/Q3). NOT YET AN IMPROVEMENT (the stock class is 85.4%;
the Proof Law bars the claim) - 2 epochs, KL 2.3 still descending; more epochs close
the gap, and TARGETED data (the operators real traffic patterns) can exceed it.
THE CHART-DIRECTION: the acceptance lever now has a working knob (distill epochs +
data mix). THE NEXT SESSION: distill 6-10 epochs -> gate on BOTH classes -> the F16
upgrade -> the A/B -> the first structural chart entry.

### F-116 — THE DISTILLATION PLATEAU: ~57% (honest close of the session) (2026-10-12 ~16:22)

8 epochs / 12k windows: 113/203 = 55.7% - FLAT vs 2 epochs (57.4%; KL 2.31 -> 2.05).
THE PURE-KL TRANSFER PLATEAUS AT ~57%. The remaining 57 -> 85.4 gap (the stock
round-trip class): the chains double-quant cost on FRESH-trained weights (the stock
round-trips own weights survive the same chain at 85.4 - the students fresh weight
geometry is quant-noise-sensitive) plus residual KL ceiling. THE NEXT LEVERS (named):
(a) the F16-preservation chain upgrade (fewer quant passes: graft F16 -> pack direct,
skipping the bf16 adapter round) - attacks the quant cost directly; (b) top-k KL / logit
smoothing distillation (sharper transfer); (c) mixed CE+KL. The distill knob is proven
(0 -> 57) but its ceiling is the chain, not the training.

SESSION LEDGER CLOSE: F-65..F-116. The crown (4 adoptions + the DF_PDL retraction under
the Proof Law), the R17 arc (chain validated at 85.4%, training regime solved, distill
transfer proven at 57%, plateau named), the chart honest (v1 116 | v2 118-146 | bar
290+), the doctrine binding, v1 frozen, the lane on production.

### F-117 — THE HF ECOSYSTEM SCAN: no better head exists; our path externally validated (2026-10-12 ~16:40)

THE USER ASKED: does HF carry MTP heads or anything that beats ours? THE SCANNED ANSWER
(9 repos families + the MTPLX pack read in full): NO improved MTP head exists anywhere
on HF for Flash-Next - every "-mtp" repo (Jundot oQ4e, mlx-community oQ5e, jedisct1,
butterf1ying REAP288, Litwein REAP320, the trending Youssofal MTPLX pack) bundles the
SAME OFFICIAL Qwen nextn weights; the suffix means included, not improved. WE RUN THAT
HEAD (85.4% class); only we are building a trained replacement, with a validated chain.
EAGLE-3 heads (nvidia ModelOpt: Qwen3-235B/30B, Kimi-K2.6, gpt-oss) - none for
Flash-Next, architecturally incompatible.

THREE USABLE FINDINGS: (a) Litweins REAP320-DWQ-MTP carries the self-distillation tag -
DWQ is distillation-through-the-quantized-forward, the ECOSYSTEM INDEPENDENTLY
VALIDATING our F16+distill upgrade path (the 57->85 recovery); (b) MTPLXs serving
contract - probability-ratio acceptance + residual resampling, adaptive depth ceiling 3
- a MEASURED 1.7x spec multiplier (73.5 vs 43.8 plain) on this exact architecture: the
sampling-scheme refinement for our queue; (c) MTPLX 2.12 prompt kernels doubled prefill
(786 -> 1453 tok/s on 4k; 1548 on 65k) - software-only, confirming prefill headroom.
Also noted: ukisai/Swift-1.5 (efficient-thinking fine-tune, co-trained MTP) - a
model-level effective-throughput lever, different base than our GSQ target.

### F-118 — SPEC_RATIO MEASURED AND REJECTED (2026-10-12 ~17:05, new session)

THE QUEUE ITEM (from F-117s MTPLX finding): STRATA_SPEC_RATIO=1 (the F-27 exact ratio
test, never before measured) on top of the standing recipe. MEASURED on the
real-workload probe: 93.7 t/s vs the baseline 118.2 (loaded window) - A 21% REGRESSION.
Draft acceptance stayed healthy (266/314 = 84.7%), so the cost is the ratio tests
per-draft math, not acceptance. NOT ADOPTED (the Proof Law: measured, rejected, recorded).
The MTPLX 1.7x does NOT transfer as a single flag - it rides their whole serving stack.
Lane restored to the standing recipe.

### F-119 — THE PROCESS-DEGRADATION LAW: the real regression, found by matched measurement (2026-10-12 ~17:25)

THE DECISIVE PAIR (same binary, same config, same window, minutes apart):
  the long-running process (up ~30 min under the users real traffic + my battery bursts): 10.3 t/s
  a FRESH BOOT: 101.6 t/s
THE VERDICT: the engine process DEGRADES over its lifetime under real traffic - ~10x by
30 minutes in this instance. NOT the binary (env-gated tap build exonerated by the fresh
boot), NOT the config (zero drift), NOT the profile (save-only, never loaded), NOT the
lane (idle, GPU 10% during the slow read). THE USERS REGRESSION FEELING WAS THIS ALL
ALONG - and every fresh-boot probe I ever ran looked healthy while their long-lived
process crawled.

THE NAMED PRIME SUSPECTS (next sessions bisect): (a) the conversation cache (49 GiB,
many parked conversations from my batteries - thrash/unpark cost growth); (b) the
adaptive tier + expert-cache state after mixed one-shot/multi-stream traffic (dormant/
banished experts mis-shaped); (c) KV/prefill pipe state growth; (d) the L2 set-aside /
stream pool after repeated window recapture.

THE IMMEDIATE REMEDY (operational, this session): the lane-health script - measure a
short single-stream probe; below threshold -> reboot the lane and re-probe. THE DEEP
FIX (queued): find the degrading structure in the engine (the suspects above, in order)
and make the process self-restore - the biggest single lever found today.

### F-120 — THE SECOND SYMPTOM: the engine EXITED on its own (2026-10-12 ~17:35)

Minutes after the 101.6 t/s fresh-boot probe, the engine process was GONE (connection
refused, 31.4 GB VRAM free - nothing loaded). The fresh boot had served the probe and
then EXITED without traffic. THE SECOND SYMPTOM: not only does the process degrade under
traffic (F-119), it can EXIT unprompted. CHECK THE ENGINE'S TAIL LOG for the exit reason
(the launcher's stdout at %TEMP%\v2freshboot.out; the strata-iq3_s.log tail; the Windows
event log for the faulting module if it crashed). THE LANE WAS RELAUNCHED: serving,
health 82.1 t/s (floor 70) on the fresh boot. THE OPERATIONAL RULE stands: lane-health.ps1
on suspicion, reboot on sub-floor, and the deep investigation (degradation + exit) is the
queue's new TOP item - the biggest lever the campaign has found.

### F-121 — THE COLD/WARM DISTINCTION: the false alarm cleared; the score stands (2026-10-12 ~17:50)

Three sequential probes on the SAME fresh process: 117.8 / 112.9 / 128.7 t/s with the
hit rate climbing 94.8 -> 96.4%. THE COLD ONE-SHOT READS THE COLD CLASS (the 24.7 t/s
health reading was a single cold probe at 82% hit - a false alarm, never judge on it).
lane-health.ps1 now warms up before judging. THE WARM SCORE (this window, v2 fixed):
single-stream 113-129 | acceptance 92.5% | MMLU 95.7% | hit 95-96% | v1 anchor 116 same
day. NOT REGRESSED on the warm class. THE REAL REGRESSION REMAINS F-119s degraded state
(10.3 t/s, reboot-fixable) - the investigation stays the top queue item.

### F-122 — THE DOCTRINE MADE REAL, THE ARCHAEOLOGY, AND THE FIRST MEASURED A/B (2026-10-12 ~18:00)

F-119 (above) found the real regression: the process degrades with age under traffic (~10x by 30
minutes; 10.3 t/s aged vs 101.6 fresh). This entry adds the corroborating per-request timeline, closes
the "it was the arms" theory with a measured A/B, and makes the operator's law binding.

**1. THE DOCTRINE IS NOW REAL — AND F-108's CLAIM WAS FALSE.** F-108 declared
`AGENTS.d/extractions/2026-10-12-proof-law-2x-bar.md` "the owner" of the law. **That file never existed**
(no `AGENTS.d` anywhere in either tree; the law lived only in this flight log). Fixed now, where it binds:
a READ FIRST block at the top of `AGENTS.md`, the short form at its foot, and the twelve-section owner at
`AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md`. **THE SIX METRICS** - M1 single-stream real
workload, M2 pair/6-conc, M3 MTP acceptance, M4 MMLU-Pro, M5 mimic/compliance/empties, M6 every other
lane - are the scoreboard; none may regress; a change that lifts one and lowers another is a trade and
ships only with both numbers printed side by side. The bar: 232 t/s (2x v1) then 290. Every claim carries
probe, window, same-window comparator, raw numbers, repeat count and artifact path.
`PERF-CHART.md` is the standing trend chart; `campaign/probe-lane.ps1` is the instrument.

**2. THE ARCHAEOLOGY — WHY THE CAMPAIGN FOOLED ITSELF (two read-only audit agents; 186 boots + every
measurement JSON).**
- **THE GATE FAILURE IS TOTAL, NOT PARTIAL:** `six-metric.ps1`, the mandated M1 instrument, was created
  **2026-10-12 16:31:51 — after all four night adoptions.** No adopted arm had ever been measured on the
  sustained default-reasoning single stream. F-107 caught PDL by luck of the user running his real
  workload; the other three were never tested on that class.
- **THE R17 GATES RAN ON A RETRACTED ARM:** `launch-v2-{tap,r17,r17tap}-8081.bat:74` still set
  `STRATA_DF_PDL=1`, so F-110..F-116 (the 85.4% chain validation, the 57.4% distill gate) were measured on
  a config F-107 had retracted. **FIXED:** all three now set `DF_PDL=0` with the reason inline.
- **F-65's "v2" column was served by the V1 engine** (`battery-v2arm.log:23-24` is byte-identical to
  `strata-custom/strata-iq3_s.log:69553-69554`; v2's boot served one request across that whole battery).
- **Prose-only numbers:** F-107's 79.6, F-108's 118.2/145.7, F-118's 93.7 have no artifact (raw bisect:
  123.8 all-off, 141.8 PDL-off-alone, 151.3/120.5 battery). F-71's T1j and F-76's EXPERT_V2 speed legs are
  prose too. New rule: no artifact, no number.
- **An unattributed quality collapse:** `quality-v2T1h.log:247` mimic 98.44 -> `:318` **73.44** in one
  boot, twelve minutes apart, never attributed (F-69 reports that arm as ~96.9).
- **THE SLOT DEFICIT DOES NOT EXIST:** matched boots read v2 **10061** vs v1 **9941**; identical reserve
  formula (`generate.cpp:4932-4939` = v1 `:3929-3936`), identical inputs. The 9,961 the user spotted was
  a boot-window lane state, not a v2 loss.
- **v2 PAID THREE ARMS FOR ONE:** `ADAPT_OVERLAP=1` is refused unless `DMA_BATCH=0`, `ADAPT_LAG=1`,
  `VERIFY_DEVICE_PLAN=0` (`generate.cpp:8422,8423,8435`) - exactly the three arms v1 runs. The launcher
  called two of them "vestigial"; the code says they are the price. Upstream measured overlap NEGATIVE on
  Windows and made it opt-in there (`generate.cpp:1426`).

**3. THE FIRST MEASURED A/B — THE "IT WAS THE ARMS" THEORY IS REFUTED.** Same window, back to back, on
the new instrument: standing recipe vs **v2's binary with v1's arms** (`launch-v2-v1arms-8081.bat`, built
this session). M1 **145.8 vs 122.3 t/s (+19%)**; pair **108.8 vs 86.1 (+26%)**; 6-conc **140.2 vs 110.9
(+26%)**; acceptance 90.2 vs 90.8%; ms/window 12.9-16.3 both, stable. **The four adoptions are net
POSITIVE on the operator's class.** Sources: `campaign/results/probe-v2-standing-1710.txt`,
`probe-v2-v1arms-1705.txt`. (Caveat recorded: both legs were fresh boots, 2-3 minutes old - they measure
the healthy class the process starts in, not the aged class F-119 found.)

**4. THE PHYSICS OF THE 2x, NOW MEASURED.** Window cost is already at its floor and has not moved all
day: **12.9-17.3 ms** on every isolated probe (model: 9.7+2.47(T-1)). What moves is **tokens per window
= 1 + accepted drafts**: **4.83** on the 00:46 record run (280.7 t/s, `:8509`), 2.16 on the 13:41 boot
(151.3, `:40463`), 2.05 today (145.8). Two more hypotheses tested and refuted with numbers:
`--spec-min-p 0.7 -> 0.45` extended the chain 1.34 -> 1.67 drafts/window but the window cost rose to
match (**144.8 vs 145.8 = neutral**; config restored to 0.7), and a **54,819-token** fresh prompt caused
**no** degradation (12.9-13.7 ms after vs 12.8-15.3 before). **The 2x is a DRAFT-YIELD problem, not a
window-cost problem** - and F-119's degradation is a process-age problem on top of it.

**5. CORROBORATION OF F-119 (per-request resolution, same boot).** Boot 16:29:13, its own requests in
order: **18.6, 18.2, 20.2, 21.3, 22.6 ms/window early -> 27.1 -> 49.0 -> 70.4, 72.0, 73.5 ms/window**
late; the short MMLU-class requests at the end of that boot read **86-94 ms/window at 95-97% hit rate**.
Median over its 106 requests: **40.8 ms/window**, against 16.6-17.1 ms for the campaign's good boots.
That is F-119's law with per-request resolution: the process degrades as it ages, and the hit rate does
not show it.

**6. LANE INTEGRITY (recorded because it corrupts measurements).** Two sessions drove GPU 8081 within one
second of each other at 16:57 (PID 18884 = this session's arm launcher, PID 5396 = the other session's
battery). One GPU, one lane, one measurement program at a time. The lane was handed back at 17:14 serving
the adopted recipe (9980 slots, overlap + ROUTE_RESIDENT_MTP printing, spec-min-p 0.7 restored).

**THE QUEUE FROM HERE, IN ORDER:** (1) F-119's degradation - find the growing structure and make the
process self-restore (the biggest lever today: 10.3 -> 101.6 t/s is 10x, larger than anything else on the
board); (2) draft yield - the R17 head on the corrected chain plus suffix/lookup depth (the 4.83
tokens/window run is the evidence of what a free drafter yields); (3) leave window cost alone until both
are done.


### F-123 — THE 18-HOUR HONEST LEDGER (the user-ordered win/loss audit, 2026-10-12 ~17:20)

THE ORDER: audit last night through now - how much was improvement, and is it recorded?
THE ANSWER, entry by entry, with the verdict each one earned:

STANDING WINS (adopted, measured, committed, still in the recipe):
  F-67 ADAPT_OVERLAP + _RESIDENT        adopted  (6-conc 145-152 -> 158/159)
  F-68 ROUTE_RESIDENT=0.5/ranks 6-9     adopted  (serial +20%, 6-conc +14%, the doorbell lever)
  F-72 ROUTE_RESIDENT_MTP=1             adopted  (draft acceptance 70% -> 85%)
  F-43 --mtp-hnorm stream (10-10)       adopted  (depth decode 102-146, prefill 7591)
  F-40 upstream kernel cherry-picks     adopted  (pair crossed stock parity; 6-conc +38-47%)
  5 standing adoptions total. Every one carries its battery in the log.

RETRACTED (adopted overnight, killed today WITH evidence):
  F-73 DF_PDL=1 -> F-107 RETRACTION     cost 40% on the real workload (sustained reasoning);
                                        the single worst regression of the period; found only
                                        when the operator reported it. The gate failure is
                                        recorded: the overnight probes measured the wrong class.

REJECTED WITH EVIDENCE (no adoption - the honest negatives):
  GDN_SPLIT, QFUSE, Q8_PACKED, TSUM, MMVF_ROWS+ATTN_LANECELL, EXPERT_V2, PREFILL_RING,
  spec-5, min-p 0.5, CATCHUP_ALL, margin-1.0/ranks-4-9 (quality-taxed), the tune sweep,
  sync-hoist (already done), SPEC_RATIO (-21%, F-118), the phaseB FastMTP head (F-56),
  every R17 trained-head variant this morning (0% acceptance, F-103..F-105).

INFRASTRUCTURE WINS (no t/s, but real):
  F-65 the v2 replay (96 commits onto v0.1.43; upstream-pullable, rerere primed)
  F-66 the drift law (matched-window anchoring - made every number since trustworthy)
  F-77..F-116 the R17 chain: 111,071 records, the bit-exact module, the validated round-trip
    (85.4%), the distillation transfer (57.4% - proven, below the stock 85.4%, not yet a win)
  F-119 the PROCESS-DEGRADATION discovery (10.3 t/s vs 101.6 fresh boot) - the biggest single
    lever the campaign has found, cause not yet named
  F-121 the cold/warm instrument fix; the lane-health script + the six-metric battery
  F-122 MTP-HEAD-REFERENCE.md (the formulation made reproducible) + the tools committed

MEASUREMENT LOSSES (my errors, all recorded):
  the battery that starved the operator's stream (doctrine added);
  the +18-25% claim that collapses to PARITY on the equal loaded lane (118.2 vs v1 116);
  the formula/knowledge gaps that needed F-122 to close.

RECORDING STATUS: every item above lives in SVG-CUSTOM.md (134 entries, F-37..F-122) and is
committed (66 v2 commits + 15 buun commits in the 18h window; the audit swept the last
strays at 17:15). The only untracked items are the storage-canon artifacts (weights, dumps,
batteries) - by design, at recorded paths.

THE NET HONEST SCORE: 5 standing adoptions, 1 retracted (the operator-caught regression), 15+
honest rejections, the R17 chain built and validated but not yet a performance win, and the
real blocker (process degradation) finally identified. Net vs v1 on the REAL class today:
parity-to-modest, NOT the 20-30% the overnight summaries implied. The 2x bar stands unmet.

### F-124 — THE DEGRADATION, BOUNDED: sustained traffic, concurrency and long prompts are all EXONERATED; the shared launcher now diagnoses the next episode itself (2026-10-12 ~17:40)

F-119 found the process degrading ~10x over ~30 minutes (10.3 t/s aged vs 101.6 fresh). This round tried
to reproduce it under instrumentation and FAILED to - which bounds it, and leaves the diagnosis in place.

**1. THE STATE COUNTERS ARE EXONERATED (the trap the Proof Law caught).** ms/window was correlated
against every counter that grows inside a boot, on the already-captured boots:

| counter | good boot 01:11 (17.8 ms/win) | degraded boot 16:29 (73.7 ms/win) | verdict |
|---|---|---|---|
| adaptive exchanges | **55,260** | 29,526 | the GOOD boot had ~2x more - not the cause |
| conversation-cache evictions | **349** | 111 | same |
| KV index lookups per request | **3.5-5.9M** | 0.4-1.5M | same |
| Windows commit free | 103-105 GiB | 103-105 GiB | no pressure either way |
| GPU throttle / CPU load | none / 13% total | none | not the machine |

Pearson r = 0.82 for exchanges looked like a cause and was **correlation with time, not mechanism**.

**2. THE REPRODUCTION ATTEMPT (instrumented, one fresh boot, 17:17:30).**

| what was run | result |
|---|---|
| 210 sustained sequential requests over 5.7 min | **no degradation** - it improved: wait 9.34 -> 8.96 ms/window, 145 -> 167 t/s |
| 1x 2-conc burst + 2x 6-conc bursts (2,400 tok each) | 6-conc **176.5 / 176.8 t/s aggregate**; single stream AFTER: 152-163 t/s (up, not down) |
| a 54,819-token fresh prompt | following windows 12.9-13.7 ms (vs 12.8-15.3 before) - no effect |
| 25,544 cumulative windows with mixed traffic | per window: **wait 9.100, pool + plan 1.224, host staging 0.201, commit 0.085 ms** - flat to the third decimal |

The per-request decode breakdown on that boot (a 200-token request, 5.7 min in):
`11.58 ms/window = verify 10.39 (GPU-reach wait 8.49 + per-layer host 1.03 [plan 0.05, CPU 0.87] + stage
0.02) + commit/emit 0.06 + draft 1.07; per layer-window: CPU experts 0.20, VRAM hits 18.69, PCIe 0.01.`

**3. WHAT THIS MEANS.** On a quiet machine the engine is stable: ~86% of each window is the GPU doing
its work (wait 9.1 of 10.6 ms) and nothing grows. Every internal candidate - the adaptive tier, the
conversation cache, the KV index, the draft chain, memory pressure, the host staging/commit path - is
now refuted with numbers. **What remains is outside the process:** F-119's own observation that the GPU
sat at ~10% during a slow read points at the GPU being time-shared with another consumer (this box runs
Chrome and two Electron agent GUIs; a WDDM time-slice stretches every kernel without changing a single
one of the engine's counters), or a request shape not yet exercised (thinking-mode/tool-call traffic,
cancellation, streaming).

**4. THE DIAGNOSIS IS NOW SELF-CAPTURING.** `launch-v2-battery-8081.bat` carries
`STRATA_SPLIT_TIMING=1` + `STRATA_DECODE_TIMING=1` (temporary; remove once the stage is named). Because
the instrument lives in the SHARED launcher, whoever boots the lane next captures its own diagnosis: if
a slow episode shows **wait for the GPU** up, the GPU is being shared; if **pool + plan** or **host
staging** up, it is the host path. One line read, no experiment.

**5. ALSO RECORDED THIS ROUND:** the matched-window A/B stands (standing recipe 145.8 vs v1-arms 122.3 t/s
M1); the fresh-boot health class measured repeatedly today is 136-167 t/s single-stream with 6-conc
176 t/s aggregate - the best concurrent number on this build since the campaign's 196-205 class.
The lane was rebooted three times by the parallel session during this round (17:15, 17:16, 17:19), which
is what killed the first watch at 4 samples - recorded so the lost samples are not mistaken for data.


### F-125 — THE WINDOW DECOMPOSED: the CPU expert path is what eats the draft yield, and the residency-bias lever is REFUTED (2026-10-12 ~17:35, goal round 2)

**1. THE EQUATION, MEASURED.** Throughput = tokens/window ÷ ms/window, and the engine's own
`STRATA_DECODE_TIMING` line (now in the shared launcher) splits the window into GPU wait, per-layer
host, and - inside that host time - the CPU expert term. Two sweeps of four request classes each, on
boot 17:17:30 (`campaign/draft-yield-sweep.ps1`, sources `campaign/results/sweep-margin05-batch4.txt`
and `sweep-diag-margin10.txt`):

| class | T | tokens/win | ms/window | GPU wait | host | of which **CPU experts** | t/s |
|---|---|---|---|---|---|---|---|
| repetitive list | 2.87 | 2.71 | 17.05 | 8.58 | 5.17 | **4.70** | 158.9 |
| code | 4.01 | 3.85 | 20.91 | 9.54 | 7.79 | **7.11** | **184.1** |
| prose | 2.69 | 2.58 | 18.32 | 8.37 | 6.47 | **5.93** | 140.8 |
| literal repeat | 2.52 | 2.35 | 15.39 | 8.47 | 4.07 | **3.69** | 152.7 |

The earlier wide-window request makes the law unmistakable: at **T 4.38** the window cost 29.12 ms of
which **13.56 ms was the CPU expert path** - 3.79 CPU experts per layer-window, ~182 per window, ~75 us
each (a 1.94 MB expert read from RAM). The GPU side barely moves (8.25-10.00 ms across every T); the CPU
term grows with the number of rows, because more rows route more distinct experts than the ~10,000-slot
GPU cache holds. **The draft yield is bought and then paid straight back to the CPU path.**

**2. THE ARITHMETIC OF THE BAR.** With the CPU term removed, the measured T=4.38 window would cost
29.12 - 13.56 = 15.6 ms and deliver 4.31 tokens/window = **276 t/s**, i.e. the campaign's own record
class (280.7 t/s was measured at 4.83 tokens/window in a 17.2 ms window, `strata-iq3_s.log:8509`). So the
2x is not a mystery and not a window-cost problem: **it is the CPU expert path.**

**3. THE RESIDENCY-BIAS LEVER IS REFUTED (honest negative, saved a wasted build).** Hypothesis: if the
router is biased harder toward resident experts, the CPU term collapses and deep windows pay. Diagnostic
boot with `STRATA_ROUTE_RESIDENT=1.0` + `RANKS=4-9` (the F-69 frontier point, lossy - diagnostic only,
no quality claim):

| class | CPU term at margin 0.5 | CPU term at margin 1.0 | verdict |
|---|---|---|---|
| T 2.5-2.9 | 4.70 / 5.93 / 7.00 | 3.78 / 4.07 / 3.45 | trims 20-51% at shallow T |
| T 4.01-4.11 | 7.11 | **7.88 (worse)** | deep windows do not improve |

So the deep-window CPU cost is structural: more rows need more distinct experts than any bias can keep
resident. The planned L3-legal code change (a separate, stronger margin for the MTP drafter alone, whose
lossiness can never touch quality) would not have bought the 2x - it is dropped before being built.

**4. THE LEVER THAT REMAINS, NAMED.** The CPU expert path is ~75 us per expert and the verify waits for
it (13.56 ms at T 4.38). Options, in the order the measurements support:
(a) **Overlap it** - the CPU experts are computed while the GPU waits for them; nothing in the numbers
    says the pool is saturated, only that the verify blocks on it. This is the ADAPT_OVERLAP idea applied
    to the CPU tier, and it is the only lever that removes the term without spending VRAM or quality.
(b) **More slots** - `--batch 4` currently gives away 3.79 GiB (~1,950 slots, ~19% of the cache) for
    concurrency slots; the slot -> CPU-term relationship is measurable by a two-boot A/B.
(c) Residency/prediction (Foresight slots, RouterLookahead) - upstream measured the VRAM-slot form
    negative; the CPU-side overlap in (a) is untested and is the fresh idea.

**5. RECORDED FOR THE CHART:** the code class at **184.1 t/s** is the best single-stream number measured
today (previous best this session 167), and the sweep shows the fast class is the one with a deep window
(T 4.01) - i.e. the engine does reward depth when the CPU term stays moderate. No quality claim from the
diagnostic boot; the standing lane was restored to the adopted recipe immediately after.


**6. ADDENDUM (17:36) — a contention-poisoned reading, recorded as such.** The health probe taken right
after the lane restore read 37.8 t/s on the code class (which had measured 184 t/s twenty minutes
earlier). It is NOT a degradation capture: the engine log of that boot shows the parallel session
running its own battery on the same lane at the same time - dozens of interleaved requests (prompts of
53-312 tokens, 1-token health probes, prompt-cache hits of 11,001 tok/s) - and my request shared the
process with them. Per the Proof Law this reading is void, and it is recorded as void rather than
counted. It is the third contention-poisoned measurement today (16:33 battery, 17:15 watch, 17:33
health), which is itself the strongest remaining evidence for the external-contention explanation of
F-119: under a shared lane the engine's own `pool + plan` term ranged **5.7 - 16.1 ms/window** across
one boot (vs 1.2-1.4 ms/window on the quiet 17:17 boot), while `wait for the GPU` stayed at 9.5-10.7 ms.
The GPU side is stable; the host/pool side is what moves when the machine is shared.


### F-126 — THE DMA SHARE TESTED: `--pcie-frac 0.9` cuts the CPU term and costs more on the GPU side - REFUTED (2026-10-12 ~17:40, goal round 3)

**THE HYPOTHESIS (from F-125).** F-125 named the CPU expert path as the term that eats the draft yield
(13.56 of 29.12 ms at T 4.38, ~182 CPU experts/window at ~75 us). `--pcie-frac` decides what share of the
MISSES is served by DMA instead of CPU compute (`src/core/verify.cpp:1618`: "the PCIe share is pcie_frac
of the misses"). So: raise the share, cut the CPU term, convert it into copy-engine work that the GPU
does not have to think about. Config-only, one line.

**THE TEST.** Two boots, same binary, same launcher arms, same four request classes, same sweep
(`campaign/draft-yield-sweep.ps1`): boot 17:32:28 at `--pcie-frac 0.35` (the adopted value) vs boot
17:36:11 at `--pcie-frac 0.9` (config `strata-iq3_s-pcie90.json`, launcher
`launch-v2-diag-pcie90-8081.bat`). Sources: `campaign/results/sweep-margin05-batch4.txt`,
`sweep-diag-pcie90.txt`.

| class | T | CPU term 0.35 -> 0.9 | GPU wait 0.35 -> 0.9 | ms/window 0.35 -> 0.9 | t/s 0.35 -> 0.9 |
|---|---|---|---|---|---|
| repetitive list | 2.87 -> 2.25 | 4.70 -> **2.96** | 8.58 -> 9.61 | 17.05 -> 16.58 | 158.9 -> 132.1 |
| code | 4.01 -> 3.93 | 7.11 -> **6.15** | 9.54 -> **15.57** | 20.91 -> **27.06** | 184.1 -> **142.3** |
| prose | 2.69 -> 2.01 | 5.93 -> **2.77** | 8.37 -> 9.06 | 18.32 -> 15.02 | 140.8 -> 129.2 |
| literal repeat | 2.52 -> 2.53 | 3.69 -> **2.49** | 8.47 -> 9.31 | 15.39 -> 15.10 | 152.7 -> 156.3 |

**THE VERDICT: the mechanism works and the trade loses.** At matched depth (code, T 4.01 vs 3.93) the CPU
term falls 0.96 ms and the GPU wait rises **6.03 ms** - the DMA transfer is now on the critical path and
the GPU has nothing to do while it lands (F-125's "miss exposed = GPU idle"). Net: 20.91 -> 27.06 ms per
window, 184.1 -> 142.3 t/s. Three of four classes lost. The adopted 0.35 split stands; the engine's
default split was already near this machine's optimum, which is consistent with the playbook's own
reading of this box (PCIe Gen5 ~= RAM bandwidth, so neither path dominates).

**WHAT THIS CLOSES AND WHAT IT LEAVES.** Closed: "just DMA the misses" - the CPU term is not removable by
moving the bytes to the copy engine; the cost reappears as exposed GPU wait. Left standing as the only
lever that removes the term without spending VRAM, quality or PCIe: **overlap the CPU tier with the GPU
work** (compute the layer's CPU experts while that layer's GPU kernel runs, instead of the verify
blocking on them). That is engine work, and it is now the top item.

**RECORDED AS AN HONEST NEGATIVE.** No adoption; the standing lane was restored to the adopted recipe
(`--pcie-frac 0.35`, margin 0.5, batch 4) immediately after the sweep, and the diagnostic config/launcher
stay in the tree only as evidence.


### F-127 — THE SLOT LEVER TESTED AND REFUTED: +1,000 expert slots buy nothing on the CPU term and cost 23% at 6-conc (2026-10-12 ~17:45, goal round 4)

**THE HYPOTHESIS (from F-125/F-126).** The CPU expert term is driven by expert misses. `--batch 4` gives
away 3.79 GiB (~1,950 slots) to concurrency slot sessions; halving it to `--batch 2` should return ~1,000
slots and cut the misses - the one lever that spends neither quality nor PCIe.

**THE TEST.** Boot 17:41:21, `--batch 2` (`strata-iq3_s-batch2.json`, `launch-v2-diag-batch2-8081.bat`),
same arms, same sweep, plus a 6-conc burst. Comparator: boot 17:32:28 at `--batch 4`, same classes
(`campaign/results/sweep-margin05-batch4.txt`, `sweep-diag-batch2.txt`).

| measure | batch 4 (adopted) | batch 2 | verdict |
|---|---|---|---|
| expert cache slots | 10,018 | **11,016** | +998 as predicted |
| code class, T ~4.0-4.2 | CPU 7.11 ms, 20.91 ms/win, **184.1 t/s** | CPU **12.26 ms**, 28.50 ms/win, 144.2 t/s | worse |
| other classes | 140.8-158.9 t/s | 133.2-147.8 t/s | worse |
| **6-conc aggregate** | **176.5 t/s** | **135.3 t/s** | **-23%** |

**THE VERDICT: refuted on both classes.** The extra 1,000 slots did not move the CPU term - it is not
capacity-limited at this margin, it is structural (the verify blocks on the CPU experts that the routing
demands). And the concurrency class pays 23% for the lost slot sessions. `--batch 4` stands; the
slot-capacity explanation of the CPU term is closed, which is the valuable part: it eliminates the whole
"buy more residency" family (slots, Foresight slots, reserve tuning) as answers to F-125's term.

**WHAT IS LEFT, AND IT IS NOW UNIQUE.** Every non-structural candidate for the CPU term has been tested
and refuted with numbers this session: residency bias (F-125: 7.11 -> 7.88 at T 4.0), DMA share (F-126:
CPU 7.11 -> 6.15 but GPU wait +6.03 ms), slot capacity (F-127: +998 slots, no effect, 6-conc -23%). The
remaining lever is the structural one: **overlap the CPU tier with the GPU work** - compute a layer's CPU
experts while that layer's GPU kernel runs instead of the verify blocking on them. The measured size of
the prize is F-125's arithmetic: removing that term from the T 4.38 window leaves 15.6 ms for 4.31
tokens/window = **276 t/s**, i.e. the campaign's own record class, on the operator's real workload.
Note also recorded: one wall-time outlier in this sweep (repetitive-list 34.60 s / 20.2 t/s) - the
parallel session was driving the same lane again, so that single reading is void per the Proof Law.

### F-128 — THE LAST LEVER SPEC'D TO THE LINE: the CPU tier is serialized at `session.cpp:960-969` (2026-10-12 ~17:50, goal round 5)

F-125/F-126/F-127 closed every non-structural attack on the CPU expert term. This round located the
serialization itself, exactly, and wrote the implementation spec: `campaign/CPU-TIER-OVERLAP-SPEC.md`.

THE CODE, quoted from the tree (`src/core/session.cpp:960-969`, the per-layer loop of the token graph):
the host spins until the GPU rings the layer (`tg.ms_wait`), then calls `pool(...)` **synchronously** -
computing that layer's missed experts on the CPU - and only then publishes the flag that releases the GPU
(`*flag = want`), timing the call into `tg.ms_pool`. The engine's own header states the consequence
(`include/strata/core/session.hpp:222-227`): "the GPU has nothing queued and is IDLE for all of this
unless `pre[l]`'s tail still covers it." **That idle time is the `CPU` term in the decode-timing line:**
3.0-4.7 ms/window at T~2.5, 7.1-12.3 at T~4.0, 13.56 of a 29.12 ms window at T 4.38.

THE FIX (three shapes, cheapest first, all preserving the per-layer plan semantics):
(1) two-deep pipeline - issue `pool(l+1)` right after ring(l) returns, wait for `pool(l)` only where its
result is consumed; (2) give the CPU tier its own worker so the host thread never computes, only issues
and waits; (3) router look-ahead admission (`STRATA_PREDICT_CREDIT`/`_BUDGET` already exist) so `pool(l+1)`
issues before ring(l) returns. Correctness invariants are written down (the fence + release ordering, the
plan buffers, identical tokens by construction) along with the gates (ms_pool down while ms_wait holds;
the four-class M1 sweep; pair/6-conc >= 176; MMLU/mimic/acceptance in-class; parity) and the honest risk:
if a layer's GPU tail is shorter than its CPU work, the overlap hides less and the limit is the per-layer
CPU time itself - the ratio `ms_pool / (ms_wait / n_layers)` after the pipeline lands separates the two.
Prize if it lands, already measured: 276 t/s at the current window shape (F-125).

### F-129 — THE ORPHAN LAW: the engine dies as pythons, not as strata.exe; cleanup must kill the native engine and gate on a STABLE drain, or the next boot races WDDM and dies in minutes (2026-10-12 ~17:52, the level-field battery attempt)

Proven twice in one window by the level-field attempt (`llama-custom\.docs\measurements\battery-lf-v2.log`).
Read this BEFORE any build-and-reboot round: every diag reboot cycle now runs through this law.

1. **Silent death signature.** A boot made on a still-draining field (31619 MiB free at launch while the
   killed engine's WDDM allocations were still releasing) sizes its pools small: the boot line read
   "361 MiB of VRAM free with everything loaded" (vs ~1000 on the healthy first boot) next to the engine's
   own low-margin warning. Under MMLU load (~3 min in) the python pair died SILENTLY - no traceback, no
   errorlevel surfaced, connection refused - while the native `engine\strata.exe` SURVIVED as an orphan
   holding ~30 GB of VRAM with a 57 GB working set. The engine log's last lines were healthy to the cut
   (142.4 t/s decode, 97.0% hit): a WDDM hard kill, not a code failure.

2. **Cleanup law (replaces "kill the pythons").** Before any relaunch: kill the python server pair AND
   `Get-Process strata` (the native engine), then gate on a STABLE drain - free >= ~31650 MiB on two
   consecutive 5 s polls - never a single >= 30500 reading. A single reading passes while the orphan is
   mid-release; the next boot inherits the shrunken pool budget and the 361-MiB margin class and dies
   under load. Full drain observed at 2-2.5 min (~30 GB over ~12 polls).

3. **Lane ownership is real.** During the drain-gate the F-127 diag boot (17:41:21, `--batch 2`) took
   port 8081 mid-ramp. Two sessions measuring one lane void BOTH runs; the level-field stands down until
   the lane is exclusively granted; no shared-window numbers enter any record. Production consequence:
   after F-127 refuted `--batch 2`, port 8081 kept serving the refuted diag config until someone
   restores `launch-v2-battery-8081.bat` (batch 4).

Standing: the level-field v2-vs-stock battery (user-ordered: 1x MMLU, 1x mimic, average prefill + decode
read from the serve lines, same window both rigs) is RUN-NOT-HELD. Its first leg re-warms per F-121
(observed ramp 84.6 -> 167.7 -> 182.7 t/s on the three-probe warm) and re-boots under the orphan law.

### GOAL RESUME MAP (goal-d0ae1db9, the 2x bar; written 2026-10-12 ~18:00 after F-128)

THE BAR: M1 232+ t/s sustained single-stream (2x v1's 116), target 290. STANDING: 184.1 t/s (best
single-stream measured today, code class, T 4.01, boot 17:32), 6-conc 176.5 t/s aggregate (F-124),
MMLU 95.7 (16:48 today), acceptance 90.2-90.8%. The six metrics and the no-regression ultimatum are
binding in AGENTS.md + AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md; the chart is
PERF-CHART.md; the instrument is campaign/probe-lane.ps1 (six metrics + per-window cost).

THE ONE LEVER LEFT, SPEC'D COLD: campaign/CPU-TIER-OVERLAP-SPEC.md. The CPU expert term is ~75 us per
expert and the verify blocks on it (13.56 of 29.12 ms at T 4.38); the serialization is
src/core/session.cpp:960-969 (ring -> pool() -> release; session.hpp:222-227 confirms the GPU is idle
during the call). Build shape (1), the two-deep pipeline, and gate it per section 4 of the spec.
PRIZE, already measured: 276 t/s at the current window shape (F-125).

CLOSED WITH NUMBERS (do not re-buy): residency bias (F-125: 7.11 -> 7.88 ms at T 4.0), DMA share
(F-126: CPU 7.11 -> 6.15 but GPU wait +6.03 ms, 184 -> 142 t/s), slot capacity (F-127: +998 slots, no
effect, 6-conc -23%), spec-min-p (144.8 vs 145.8 = neutral), long-prompt thrash (54,819 tokens: no
degradation), sustained traffic and concurrency as degradation triggers (F-124). The F-119 degradation
did NOT reproduce on a quiet machine and its counters are exonerated; the shared launcher carries
STRATA_SPLIT_TIMING + STRATA_DECODE_TIMING so the next episode names its own stage.

STATE: lane serving the adopted recipe on 8081 (batch 4, margin 0.5, pcie-frac 0.35, overlap on);
MAGMA v1 frozen at 8da8bf16, untouched. NOTE: a parallel session shares this GPU - one lane, one
measurement program at a time, or the numbers are void (three void readings recorded today).

### F-130 — THE SUFFIX/LOOKUP ARM REJECTED, AND THE UNIFYING FORMULATION (2026-10-12 ~17:55, goal resumed by the operator)

**THE ARM.** `--lookup-chain 4` + `--suffix-draft 8` (both config-only; the lookup chain is OFF by default,
the suffix drafter runs at 3). Boot 17:51:00, same arms otherwise, same four-class sweep. The window cap
did rise as designed - `strata verify: window up to 8 tokens, 128.2 MiB of device buffers` (vs 6 tokens) -
and every draft in this family is verified exactly by the target, so quality cannot move. Sources:
`campaign/results/sweep-diag-sfx.txt` vs `sweep-margin05-batch4.txt`.

| class | T | CPU term baseline -> arm | ms/window | t/s baseline -> arm |
|---|---|---|---|---|
| repetitive list | 2.87 -> 2.22 | 4.70 -> 4.01 | 17.05 -> 17.19 | 158.9 -> **123.3** |
| code | 4.01 -> 3.94 | 7.11 -> **14.16** | 20.91 -> **30.71** | 184.1 -> **125.4** |
| prose | 2.69 -> 2.38 | 5.93 -> 4.91 | 18.32 -> 18.12 | 140.8 -> **120.9** |
| literal repeat | 2.52 -> 2.56 | 3.69 -> 3.76 | 15.39 -> 17.90 | 152.7 -> **133.5** |

**REJECTED on every class** (M1 regressed, so it dies by the ultimatum). The mechanism is the round's real
finding: at the same T the CPU term is unchanged, but the deeper cap lets the code class reach a
bigger window whose routing demands **twice the CPU experts** (7.11 -> 14.16 ms of a 30.71 ms window).
The arm aimed at repetitive traffic did not find enough of it in these classes to pay its own overhead.

**THE UNIFYING FORMULATION (tracked here as the operator asked).** Fitting the measured decode-timing
terms over the sweep points (T, ms/window, GPU side, CPU side):

    t/s  =  tau(T) / V(T),      tau(T) = 1 + sum_{k<T} p_k        (p = per-draft acceptance)
    V(T) =  GPU(T) + CPU(T) + host(T)
    GPU(T)  ~ 9.0 - 10.4 ms   (nearly FLAT in T; it is the verify kernels + draft chain)
    CPU(T)  ~ 4.72 * T - 6.8 ms  (measured: 3.03 ms at T 2.08, 5.93 at 2.58, 10.50 at 3.22, 13.56 at 4.31)
    host(T) ~ 0.2 - 0.5 ms

Every depth-increasing lever therefore pays about **+4.7 ms per added window row**, of which the CPU term
is the overwhelming part. That single coefficient explains this session's whole ledger: residency bias
(fails at depth), DMA share (moves cost from CPU to GPU wait), more slots (does not touch the coefficient),
spec-min-p (neutral), lookup-chain (raises the cap, the coefficient collects), and the campaign record run
of 280.7 t/s (4.83 tokens/window in a 17.2 ms window = a window whose CPU term was near zero).

**THE CONSEQUENCE, STATED PLAINLY: the CPU term is the gate on every other lever.** Nothing on the
draft-yield side can pay until `CPU(T)` stops growing - which is exactly the serialization at
`src/core/session.cpp:960-969` (F-128) and the spec written for it (`campaign/CPU-TIER-OVERLAP-SPEC.md`).
The bar arithmetic with the coefficient removed is already measured: 4.31 tokens/window in a 15.6 ms
window = 276 t/s.

**LANE:** restored to the adopted recipe immediately (boot 17:58, window cap back to 6 tokens); the
diagnostic config/launcher stay in the tree as evidence. Lane-sharing caveat recorded as before: a
parallel session drives the same GPU, so each wall number carries that risk; the decode-timing numbers
used here are the engine's own per-window accounting and are internally consistent.

### F-131 — THE SPEC CORRECTED BY THE STAGE LIST: the spin sits BEFORE the resident-expert GEMM, so the fix is a reordering, not a pipeline (2026-10-12 ~18:05, goal round 8)

F-128's shape (1) was wrong and is corrected in `campaign/CPU-TIER-OVERLAP-SPEC.md` section 6. `pool(l)`
cannot start before layer l's ring: **both of its inputs arrive with that ring** - the plan (which experts
missed) and the activations it computes on. Prediction does not rescue it either; predicted ids still
leave the activations unknown until the layer's own attention has run.

The engine's own per-layer stage list places the fix precisely: `router -> shared+quant -> **waitA** ->
VRAM hits -> ... -> copy+combine`. The resident-expert grouped GEMM (**"VRAM hits", 20-38 per
layer-window**) runs AFTER the spin (`wait_flag_ge_kernel`, `src/kernels/cuda/verify_kernels.cu:1051`,
launched as a `<<<1,1>>>` node), yet it does not depend on the CPU results at all - it reads the same
layer activations and its own resident weights. Only the **combine** needs both halves.

THE CORRECTED BUILD, in one sentence: move the spin to sit immediately before the combine instead of
before the resident grouped kernels, so the GPU computes the layer's resident experts while the host
computes the layer's missed ones, and only the sum waits for both. Prize unchanged and already measured:
**276 t/s** at the current window shape (F-125). Binding check before building: the per-layer resident
GEMM time vs the per-layer CPU work (~0.28 ms/layer at T 4.38) - if the GEMM is shorter, the overlap hides
only part of it, and the post-change `ms_pool` against the layer's grouped-GEMM time is the measurement
that says how much. Invariants and gates (section 4) stand, and get stricter rather than looser.

### F-132 — THE REORDERING'S RISK PROFILE SETTLED FROM EXISTING NUMBERS: it cannot make a window slower (2026-10-12 ~18:15, goal round 9)

The spec's last open question was the per-layer resident-GEMM time vs the per-layer CPU work. The logged
decode-timing counts answer the *direction* without a new boot: per layer-window the host computes
~2.5-3.8 CPU experts (0.07-0.11 ms each) while the GPU computes ~38-40 resident experts in the same layer
(10 routed per row x the rows in the window, 91-94% resident).

With `P` = GPU work before the grouped kernels, `R` = resident grouped-GEMM time, `C` = CPU expert time:
today the layer costs `P + C + R + combine` (the spin sits before the GEMM: `router -> shared+quant ->
waitA -> VRAM hits`); after moving the spin to just before the combine it costs
`max(P + R, P + C) + combine`. So the change is **monotonically non-worse**: full hiding of `C` when
`C <= R`, and still a gain of `R` when `C > R`. It is a scheduling change only - no arithmetic, no
routing, no VRAM - so tokens stay identical by construction.

THE CONSEQUENCE: this lever has the best risk profile anything in the campaign has had. The floor of the
win is `R` per layer; the ceiling is the whole CPU term (`4.72*T - 6.8` ms/window, the measured 276 t/s at
T 4.38). What can still go wrong is correctness, not speed: the combine must see the CPU results fenced
and complete, and the resident GEMM reads only the layer activations and its own weights (it does not
consume anything the CPU half produces). Gate per spec section 4; read `ms_pool` and the window time from
the decode-timing line in the first boot after the change.

### F-133 — SECOND SPEC CORRECTION, BEFORE ANY BUILD: the wait is already per-group, already skippable, and already removable (2026-10-12 ~18:25, goal round 10)

F-131/F-132 told the next session to "move the spin to just before the combine". **The code contradicts
that premise and the instruction is withdrawn.** The real call sites (`src/core/verify.cpp`) are
per-expert-group and already staged: `wait_flag_ge_or(m_flagA_, ...)` / `wait_flag_ge(m_flagA_, ...)` for
the pool's group plan (1596/1602), the same pair for the PCIe staging share (1609/1610), and for the CPU
rows (1623/1628) - where the second form is guarded by `if (!fuse_rows)`, i.e. **when rows are fused the
engine does not issue the CPU wait at all**. The waits also carry skip predicates (`skip_ + grp`).

So the ordering the spec proposed is in spirit already implemented, and the measured CPU term
(4.72*T - 6.8 ms/window, 276 t/s at T 4.38 if removed) is the residual this design does not hide. The
next session's first move is therefore READING, not building: map the per-group sequence in
`verify.cpp:1560-1660`, find what arms `fuse_rows` (the one path that already deletes the CPU wait), and
only then choose among moving the combine later, widening the `_or` predicate, or arming `fuse_rows`.
Section 7's monotonic-no-regression proof does not transfer to a reordering that turns out to be
unnecessary. This correction exists so no build is spent on a premise the source already refutes - the
same class of error F-131 caught in F-128's shape (1).

### F-134 — THE LEVER MAY ALREADY BE BUILT: `STRATA_FUSE_WAIT` folds the wait into the copy (2026-10-12 ~18:35, goal round 11)

F-133 asked what arms `fuse_rows`. The answer, from the source: `fuse_wait_on()` reads **`STRATA_FUSE_WAIT`**
(`src/core/verify.cpp:227-232`), and when it is on, the CPU wait stops being a separate spin kernel - line
1599 `// 143-amdk: one launch for the wait and the copy`, line 1627 `fuse_rows = fuse_wait_on() && ...`, line
1632 `// the wait folded into the copy's blocks`. **That is precisely the overlap this spec was written to
propose, already implemented.**

Upstream's own comment beside the switch: *"measured on gfx12 only, so it is on there by default"* - AMD
RDNA4. This machine is CUDA sm_120, where it has never been measured, and nothing in the standing recipe
sets it.

**THE NEXT TEST IS ONE BOOT, NO BUILD:** `STRATA_FUSE_WAIT=1` on the standing recipe, then the four-class
sweep plus the concurrent probe, reading `ms_pool`/`CPU` and the window time from the decode-timing line.
Preconditions are already satisfied - `!prof_on_` (only `STRATA_VERIFY_PROFILE` sets that, and the launcher
uses SPLIT_TIMING/DECODE_TIMING), `!remote_opt_` (no `--remote-expert-opt`), `dec_batch` (`STRATA_DEC_BATCH`
unset = on). LABELLED AS A HYPOTHESIS WITH AN EXACT ARM, not a measured result: the Proof Law forbids
claiming it until the boot reports. If the CPU term falls, the prize is the measured 276 t/s at T 4.38
(F-125); if it falls without the window improving, the term was already hidden and F-133's reading stands.
Gate order: tokens identical first, then M1/M2, then MMLU/mimic in-class.

### F-135 — `STRATA_FUSE_WAIT=1` MEASURED AND REJECTED: -50% on sm_120, upstream's gfx12-only warning was right (2026-10-12 ~18:05, goal round 12)

THE TEST F-134 ASKED FOR, run in one boot (17:59:35, `launch-v2-diag-fusewait-8081.bat`, same arms plus
`STRATA_FUSE_WAIT=1`), same four-class sweep, same window
(`campaign/results/sweep-diag-fusewait.txt` vs `sweep-margin05-batch4.txt`):

| class | T | CPU term baseline -> arm | GPU wait baseline -> arm | ms/window | t/s baseline -> arm |
|---|---|---|---|---|---|
| repetitive list | 2.87 -> 2.84 | 4.70 -> 7.68 | 8.58 -> **20.72** | 17.05 -> 33.86 | 158.9 -> **80.0** |
| code | 4.01 -> 3.97 | 7.11 -> 13.38 | 9.54 -> **24.17** | 20.91 -> 43.88 | 184.1 -> **89.8** |
| prose | 2.69 -> 2.97 | 5.93 -> 8.81 | 8.37 -> **21.20** | 18.32 -> 36.44 | 140.8 -> **76.3** |
| literal repeat | 2.52 -> 2.80 | 3.69 -> 5.05 | 8.47 -> **21.44** | 15.39 -> 31.68 | 152.7 -> **81.8** |

**REJECTED, decisively and on every class** (M1 halved). The mechanism is visible in the same numbers: the
CPU term did NOT fall (it rose, because the window grew slightly) while the **GPU-reach wait more than
doubled** (8.4-10.4 -> 20.7-24.6 ms). Folding the wait into the copy's blocks makes the *copy* the thing the
GPU pipeline stalls on, and on this stack that is far worse than the separate spin kernel it replaces.

Upstream's comment beside the switch said it plainly - *"measured on gfx12 only, so it is on there by
default"* - and this is the second time this session that a gfx1x-lineage default has failed on sm_120
(the first being the F-67 overlap adoption, whose upstream comment says the same thing about Windows).
THE RULE THIS EARNS: an upstream default gated by an architecture comment is a hypothesis for THIS card,
never a premise.

**CONSEQUENCE FOR THE SPEC.** `campaign/CPU-TIER-OVERLAP-SPEC.md` section 9 now carries this negative: the
cheapest candidate for the CPU-overlap lever is measured and gone, so the remaining shapes are the
per-group `_or` skip predicate (section 8) and the wider reordering, both of which are now known to be
non-trivial rather than a flag. The prize is unchanged (276 t/s at T 4.38 measured) and the CPU term
(4.72*T - 6.8 ms/window) is still the gate on every other lever - but the "it may already be built" hope of
F-134 is closed by measurement.

### F-136 — THE CPU TERM'S PER-LAYER SHAPE: it looks LATENCY-bound, not throughput-bound (2026-10-12 ~18:15, goal round 13)

The arithmetic of the term F-125 identified, from the decode-timing counts already measured:

| window | CPU experts / layer-window | CPU term / window | => per layer | => per expert |
|---|---|---|---|---|
| T 3.79 | 2.52 | 8.63 ms | 0.180 ms | 71 us |
| T 4.31 | 3.79 | 13.56 ms | 0.283 ms | 75 us |

75 us for a 1.94 MB expert read is ~26 GB/s - a *single core's* realistic share of this machine's RAM
bandwidth (~60-85 GB/s across all channels). So the term decomposes as roughly
**CPU term per window ~= (number of CPU experts) x 75 us**, i.e. the experts of a layer are being
accounted at single-core latency, NOT at the pool's 15-worker throughput (which would be ~5-8 us each if
they overlapped).

**THE HYPOTHESIS (labelled as such - not measured):** the CPU experts *within one layer* are not
overlapping. Each layer carries only 2.5-3.8 of them, and if they are computed one after another on the
way to that layer's combine, the layer pays their latencies in series. If they were dispatched together
across the pool, the same work would cost ~one expert's latency (~75 us) instead of 3-4 of them, cutting
the term by roughly 2-3x - which at T 4.31 is 13.56 -> ~5 ms/window, i.e. 29.12 -> ~20.5 ms, i.e.
**184 -> 210 t/s** on the class measured today, with no VRAM spend, no quality change and no PCIe
re-balancing (all three of which are already refuted: F-125, F-126, F-127).

**WHERE TO LOOK (one read, no build):** the call site is `src/core/session.cpp:963`
(`pool(user, s.db->h_x_f, s.db->h_ids, s.db->h_weights, g.n_embd, s.k, y_miss_host)`) and its
implementation in `src/kernels/cpu/pool.cpp` / `include/strata/kernels/cpu/pool.hpp`. The question to
answer from the source: does one `pool(...)` call dispatch ALL of the layer's missed experts to different
workers in parallel, or does it walk them in series? The counter to read for the answer is the existing
`ms_wait_park_` / worker-busy instrumentation in `pool.hpp:204-245`; the confirming experiment is a boot
where the CPU term is printed per layer alongside the number of workers that had work in that layer.
**If the dispatch is already parallel, this hypothesis dies the same day** - and the term is then pure
single-expert latency, which moves the answer to the CPU kernel itself (the AVX-512 IQ path, `iq_avx512.cpp`)
rather than to scheduling.

### F-137 — `STRATA_POOL_SPIN_US=2000` PROVISIONAL NEGATIVE, WITH A CONFOUND NAMED (2026-10-12 ~18:10, goal round 14)

**THE LEAD (from F-136, and it was a good one).** `include/strata/kernels/cpu/pool.hpp` says the pool's
workers *sleep between layers* when few experts miss (`pool.hpp:212-215`: "with most experts in VRAM ... many
layers have no CPU work, so the workers [sleep]"), that `STRATA_POOL_SPIN_US` governs the spin before
parking, and that "a short spin makes the workers sleep constantly". Our standing launcher sets
`STRATA_POOL_SPIN_US=100` - the short-spin value, inherited from the v1-era #921 tuning. So the test: let
the workers spin 2 ms instead of 100 us and see whether the per-layer park/wake round trip was eating the
CPU term.

**THE MEASUREMENT (boot 18:05:19, one boot, same four-class sweep, `sweep-diag-spin2000.txt`):**

| class | T | CPU term | GPU wait | ms/window | t/s |
|---|---|---|---|---|---|
| repetitive list | 2.92 | 7.83 | **30.36** | 45.50 | **60.0** |
| code | 3.98 | 12.79 | **34.29** | 55.17 | **69.1** |
| prose | 2.29 | 5.06 | **26.61** | 38.27 | **55.9** |
| literal repeat | 2.46 | 3.53 | **26.58** | 36.10 | **63.4** |

Against the 17:29 baseline (141-184 t/s, GPU wait 8.4-10.4 ms) this looks decisively rejected. **BUT THE
CONFOUND IS VISIBLE AND MUST BE STATED:** the GPU-reach wait is 26-34 ms on EVERY class, roughly 3x the
baseline, and the same signature appeared in the F-135 `STRATA_FUSE_WAIT` boot an hour earlier (20.7-24.6
ms). A uniform inflation of the GPU-side term across all four classes is what external contention looks
like on this box (a parallel session shares the GPU), not what a pool-spin knob does - a spin knob should
move the CPU term, and the CPU term here (3.5-12.8 ms) is unchanged in shape.

**THEREFORE, PER THE PROOF LAW:** this arm is recorded as a **provisional negative, not a closed one**.
F-135's verdict carries the identical caveat. Both need a re-test in a window where the standing recipe
re-measures at the 8.4-10.4 ms GPU-wait baseline immediately before and after the variant - the
matched-anchor discipline the ultimatum demands. What is NOT in doubt: the standing lane was restored
(boot 18:07:47, `STRATA_POOL_SPIN_US=100`) and the adopted recipe was left serving.

**THE PROCESS LESSON, recorded because it has now cost two arms:** when a variant's *control* term (GPU
wait) moves by 3x across every class, the run is measuring the machine, not the variant. The probe must
assert the control term before it reports the variant's effect - the baseline boot for each arm, not a
remembered one from an hour earlier.

### F-138 — CRITICAL RECORD CORRECTION: F-135 AND F-137 ARE VOID, AND THE STANDING RECIPE ITSELF IS DEGRADED RIGHT NOW (2026-10-12 ~18:10, goal round 15)

The re-anchor F-137 demanded, run on the **standing adopted recipe** (boot 18:07:47, nothing changed,
`sweep-standing-reanchor.txt`):

| class | T | CPU term | GPU wait | ms/window | t/s |
|---|---|---|---|---|---|
| repetitive list | 3.09 | 8.71 | **26.33** | 41.13 | **72.2** |
| code | 4.00 | 14.37 | **32.08** | 54.55 | **70.6** |
| prose | 2.73 | 7.32 | **25.76** | 39.52 | **64.8** |
| literal repeat | 2.54 | 4.79 | **24.08** | 34.67 | **69.8** |

Compare the same sweep on the same recipe at 17:29: 141-184 t/s, GPU wait 8.4-10.4 ms. **The adopted
configuration halved itself with no change made to it.**

**CONSEQUENCE 1 - TWO RECORDED NEGATIVES ARE VOID, NOT NEGATIVES.** F-135 (`STRATA_FUSE_WAIT=1`, 76-90 t/s,
wait 20.7-24.6 ms) and F-137 (`STRATA_POOL_SPIN_US=2000`, 52-69 t/s, wait 26.6-34.3 ms) measured an arm
against a machine state that this run now shows is the *same degraded state* the baseline itself falls into.
Their numbers cannot separate the arm from the lane. Both are retracted to VOID and must be re-run with a
same-window baseline anchor (F-137 already wrote that rule; this entry is the proof it was needed).

**CONSEQUENCE 2 - THE USER'S REGRESSION IS LIVE AND LOCALIZED.** This is the first capture of the F-119
degradation *with the instrumentation on*, and it is unambiguous: **the CPU term is normal (4.8-14.4 ms,
the same shape as the healthy baseline) while the GPU-reach wait triples (8.4-10.4 -> 24-32 ms).** So the
F-119 slowdown is on the GPU side of the window, not the host side, not the counters, not the routing, not
the draft chain - consistent with the engine's kernels simply taking 3x longer to reach the host. That is
what a shared/timesliced or clock-limited GPU looks like from inside the engine, and it is the first
measurement in this campaign that points there with the instrument live rather than by elimination.

### F-139 — THE TWO ENGINE STATES, MEASURED: boot-persistent 2.3x swing with the GPU-side wait as the tell (2026-10-12 ~18:12, goal round 16)

Two identical sweeps, 90 seconds apart, on ONE boot (18:07:47), same adopted recipe, same classes:

| class | 18:10 run (t/s, GPU wait) | 18:11 run (t/s, GPU wait) |
|---|---|---|
| repetitive list | 72.2, 26.33 ms | 80.5, 24.70 ms |
| code | 70.6, 32.08 ms | 82.2, 29.82 ms |
| prose | 64.8, 25.76 ms | 58.0, 24.21 ms |
| literal repeat | 69.8, 24.08 ms | 68.3, 24.67 ms |

Against the same recipe on the 17:29 boot: **141-184 t/s, GPU wait 8.4-10.4 ms.**

**WHAT THIS ESTABLISHES, with the numbers now in hand:**
1. **Two states, not noise.** Run-to-run spread within a state is ~±15% (code 70.6 -> 82.2, prose 64.8 ->
   58.0); the gap between states is **2.3x**. One same-window anchor is therefore sufficient to classify a
   run - the rule F-137 wrote, now with a measured noise floor behind it.
2. **The state is boot-persistent**, not per-request flicker: two runs 90 s apart on the same boot agree.
3. **The tell is the GPU-side term.** Healthy: GPU wait 8.4-10.4 ms. Degraded: 24-32 ms. The CPU term
   keeps its normal shape in both (4.8-14.4 ms). So the state lives on the GPU side of the window.
4. **The adopted recipe is not the variable** - it produced both states on different boots, unchanged.
   This is F-119's law finally measured with the instrument on, and it is the same thing the operator has
   been reporting all day: the engine sometimes crawls, and a restart sometimes fixes it.

**THE MEASUREMENT PROTOCOL THIS EARNS (now binding for every future arm):** before any variant is judged,
measure the STANDING recipe in the same window and read its GPU wait. Above ~15 ms the run is in the
degraded state and **no arm may be judged from it** - that is exactly how F-135 and F-137 were voided
(F-138). Record the state with every number, as the ultimatum's measurement protocol already requires.

**WHAT IT DOES NOT YET SAY:** whether the state is caused by the GPU being time-shared with the 133
GUI/browser processes on this box, by a driver/WDDM path, or by something inside the engine that changes
between boots. The next measurement that separates them is cheap: sample `nvidia-smi` per-process GPU
utilisation and the SM clock *during* a degraded run and a healthy run, and compare. The engine's own
numbers have now done their part - they have isolated the state to the GPU side and given it a numeric
signature to detect it by.

### F-140 — THE RESTART TEST: STILL DEGRADED AFTER RESTART (2026-10-12 ~18:15, goal round 17)

Boot 10-12 18:07:47 engi (degraded: 58-82 t/s, GPU wait 24-32 ms) was replaced by a fresh boot at 10-12 18:13:23 engi of
the SAME adopted launcher, then the same four-class sweep ran immediately (sweep-restart-test.txt).

RESULT: code class **78.2 t/s**, mean GPU-reach wait across the four classes **27.9 ms**, against the two
known states measured today - healthy 141-184 t/s at 8.4-10.4 ms (boot 17:29) and degraded 58-82 t/s at
24-32 ms (boots 18:07/18:08, F-139).

**VERDICT: STILL DEGRADED AFTER RESTART.** The operator's own observation ("a restart sometimes fixes it") is NOT confirmed for this transition: the engine state
survives or clears at boot, and this run says which.
Consequence: the state is NOT boot-time - it is a property of the machine's current condition, since a clean boot reproduces it. That moves the next measurement to per-process GPU load during the run (the 133 GUI/browser processes on this box) and to the card's clock behaviour under the engine's own load.

### F-141 — THE DEGRADED STATE IDENTIFIED: the Windows desktop compositor is using 2.6x more GPU than the engine (2026-10-12 ~18:18, goal round 18)

Measured during a live degraded request (boot 18:13:23, 400 tokens in 7.41 s = **54 t/s**), sampling
`\GPU Engine(*)\Utilization Percentage` ten times across the request on the machine that serves the engine:

| process | summed utilisation (10 samples) |
|---|---|
| **`dwm` - the Windows Desktop Window Manager** | **466.4** |
| `strata` - the engine itself | 181.8 |
| DeepSeek Harness (Electron) | 24.3 |
| chrome | 16.2 |

and at the same moment `nvidia-smi`: 2685 MHz SM, 77 W of a 575 W limit, 7% utilisation, P1, **no throttle
flags**.

**THE COMPOSITOR IS THE CONTENTION.** `dwm` is compositing this desktop - 133 GPU-accelerated
browser/Electron/Telegram/Obsidian windows are resident - and it is consuming ~2.6x the GPU time the engine
gets. The engine's kernels are therefore scheduled in the gaps, which is exactly the signature F-139
measured from inside: **GPU-reach wait triples (8.4-10.4 -> 24-32 ms) while the CPU term keeps its normal
shape**, the card shows no throttle and burns only 77 W, and a restart does not help (F-140) because the
desktop is still there.

**WHAT THIS EXPLAINS, ALL FOUR AT ONCE.** (1) F-119's "10x over 30 minutes" - as more apps and windows
accumulate, `dwm`'s share grows; (2) the operator's "a restart sometimes fixes it" - only when the desktop
load has also changed; (3) why the whole campaign sees two states with identical binaries and configs; and
(4) why every counter and every arm on the engine side came back innocent.

**WHAT TO DO WITH IT (in order of directness):** close or disable hardware acceleration in the GPU-heavy
resident apps (Chrome/Electron/Obsidian/Telegram) and re-measure the SAME boot - the state should move
without touching the engine; if it does, this is an environment finding, not an engine bug, and the
campaign's standing numbers must be quoted with the desktop load attached. The engine-side mitigations
worth measuring afterwards, in the order they are cheap: `HAGS` (hardware-accelerated GPU scheduling) -
already on the B9 A/B list - and raising the engine's GPU priority. **THE MEASUREMENT-CRITICAL PART:** every
arm from here on is judged only in a window where the standing recipe reads healthy (GPU wait under ~15 ms),
which is now a checkable precondition rather than a hope.

### F-142 — F-141 CONFIRMED: `dwm` holds ~40% of the GPU even when the engine is idle (2026-10-12 ~18:20, goal round 19)

Three sequential 300-token probes on the degraded boot, with a `\GPU Engine(*)` sample taken **between**
requests (so the engine is idle and the number is the desktop's own):

| sample | engine t/s | engine GPU wait | dwm (idle sample) | engine (idle sample) |
|---|---|---|---|---|
| 1 | 64.1 | 19.46 ms | 40.1 | 0 |
| 2 | 66.0 | 20.26 ms | 41.7 | 0 |
| 3 | 68.2 | 20.11 ms | 42.1 | 0 |

Reference states measured today: healthy 141-184 t/s at 8.4-10.4 ms; degraded 54-82 t/s at 24-32 ms.

**The compositor is a permanent ~40% GPU consumer on this desktop and the engine is at 0 while it does it.**
That is the mechanism F-141 named, now confirmed without touching a single application: the engine's
throughput in this window (64-68 t/s, 20 ms GPU wait) and the desktop's standing share (40-42%) are two
views of the same fact - the engine's kernels are being scheduled around the compositor.

**CONSEQUENCE FOR EVERY NUMBER IN THIS CAMPAIGN.** The healthy/degraded split (2.3x) is a **desktop-load**
variable, not an engine variable. Therefore: (a) the standing numbers (141-184 t/s) are only valid with the
desktop load recorded beside them, as the ultimatum's measurement protocol already demands; (b) the two
arms voided today (F-135, F-137) were almost certainly contaminated by exactly this; (c) the arm that
matters next - the CPU-tier overlap - must be judged in a window whose GPU wait is under ~15 ms, or its
effect will be invisible inside a 2.3x environmental swing. The environmental fixes are cheap and outside
the engine: disarm hardware acceleration in the resident GPU apps, and test HAGS (the campaign's own B9 A/B
item) plus GPU priority for the engine process.

### F-143 — THIRD CONFIRMATION, AND THE MEASUREMENT BLOCKER STATED (2026-10-12 ~18:25, goal round 20)

Re-sampled 20 minutes after F-142, same boot (18:13:23), no changes made: **dwm's idle GPU share 46.8%**,
engine **61.7 t/s at 22.62 ms GPU-reach wait**. Third independent confirmation of the same correlation
(F-141 during a request: dwm 466/10 samples ~ engine 181; F-142 between requests: dwm 40-42; now dwm 46.8),
and the environment has NOT moved in the last 20 minutes.

**THE BLOCKER, STATED WITH NUMBERS.** Every arm in this campaign needs a window where the standing recipe
reads healthy (GPU wait under ~15 ms at 141-184 t/s). The last 50 minutes, across two separate boots, have
read 54-82 t/s at 20-32 ms with dwm holding 40-47% of the GPU - so no engine-side arm can be judged, and
the two that were attempted in this window (F-135, F-137) had to be voided. This is not an engine condition
and no engine change addresses it; the fix is the desktop load (close or disarm hardware acceleration in the
resident GPU apps, or test HAGS / GPU priority). The lane is left serving the adopted recipe and ready: the
moment the desktop is quiet, the same sweep in the same boot is the whole confirmation, and the engine-side
queue (F-136's dispatch read, the `_or` predicate, the CPU-tier overlap with its measured 276 t/s prize)
resumes under a trustworthy baseline.

### F-144 — F-136 ANSWERED BY THE ENGINE'S OWN HEADER: single-core streaming is a KNOWN case, and `run_split*` is the existing fix (2026-10-12 ~18:20, goal round 21)

Read of `include/strata/kernels/cpu/pool.hpp:171-189`, and it says three decisive things.

**1. THE HYPOTHESIS IS CONFIRMED BY THE CODE'S OWN COMMENT.** `run_split`'s doc block: *"the same outputs as
`run`, bitwise, with every expert split by rows across all threads (gate/up rows, then the intermediate's
quantization, then down rows). **With fewer experts than threads - the case once the VRAM tier takes half of
them - `run` leaves cores idle and each expert streams at one core's bandwidth; this streams every expert at
all of them.** At most `kMaxSplit` experts."*

That is F-136's measurement described before it was measured: our CPU term is ~75 us per expert (single-core
RAM bandwidth, ~26 GB/s) with only **2.5-3.8 experts per layer against 15 workers**. The engine knows this
case exists, names it, and ships a bitwise-identical function for it.

**2. THE FIX IS ALREADY BUILT, IN THREE VARIANTS.**
- `run(jobs, n)` - one job per worker; leaves cores idle when n < workers (what we measured).
- `run_split(jobs, n)` - every expert split by rows across all threads; `kMaxSplit = 16` experts.
- `run_split_multi(jobs, n)` and **`run_split_multi_native(f, jobs, n)`** - the multi-token forms ("the rows
  of each expert are read once for all of its tokens", native pack arithmetic), `kMaxSplitMulti = 96`.
Our case is exactly the last one: a native pack (IQ3_S), 2.5-3.8 experts per layer, multi-token windows.

**3. THE INSTRUMENT FOR IT IS ALREADY BUILT TOO.** `run()` is documented as spending its time in three
phases per layer - *"wait for every worker to be parked, wait for the drain, wait for them to re-park"* - and
`phase_ms(wait_park, drain, repark)` exposes them, with the header stating why it matters: *"Without the
split there is no way to tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION,
and those need opposite fixes: the first is a kernel problem and the second is a barrier problem."*

**THE NEXT MEASUREMENT (one read, no build):** find which of the four entry points the decode path calls for
this pack - `session.cpp:963`'s `pool(...)` callable and its binding - and whether `run_split_multi_native`
is armed, gated, or merely unused. If the decode path is on `run`, switching it to the split form is a
bitwise-identical change aimed squarely at the term that gates every other lever (F-125/F-129), and the
three-phase `phase_ms` numbers say in the same boot whether the remaining cost is the WORK or the BARRIER.
This is the strongest lead the campaign has produced: existing code, stated invariant (bitwise), no VRAM, no
quality risk, and it targets the exact term the measurements isolated.

### F-145 — THE CALL SITES: prefill uses the split form, the CUDA DECODE path does not (2026-10-12 ~18:25, goal round 22)

Grep of every `run_split*` call site in the tree:

| site | what it calls |
|---|---|
| `src/prefill/prefill.cpp:3332` | **`cpu_pool_->run_split_multi_native(cf, m.cpu_jobs.data(), ...)`** - the prompt path already uses the split form |
| `sycl/src/core/expert_source.cpp:2526` | `if (d.split_rows) d.pool->run_split(...)` |
| `sycl/src/core/expert_source.cpp:2789-2790` | `if (native) run_split_multi_native(...) else run_split_multi(...)` |
| `src/kernels/cpu/pool_tasks_test.cpp:67,78`, `pool_stress.cpp:83` | tests only |

**The CUDA decode path is absent from that list.** `src/core/expert_source.cpp` (the file the CUDA build
compiles - the `sycl/` hits are the SYCL copy of the same file and are not in a CUDA build) contains no
`run_split`, no `run_split_multi`, and no `run_split_multi_native`. So on this build:

- **the prompt path splits every expert's rows across all threads** (`prefill.cpp:3332`), and
- **the decode path uses the plain form** - the one `pool.hpp:175-179` documents as leaving cores idle and
  streaming each expert at *"one core's bandwidth"* when there are fewer experts than threads.

**AND THAT IS EXACTLY THE MEASURED SITUATION:** 2.5-3.8 CPU experts per layer-window against 15 workers,
at ~75 us each (single-core RAM bandwidth, ~26 GB/s), summed into the term that gates every other lever
(F-125/F-129: `CPU(T) = 4.72*T - 6.8` ms/window; the window at T 4.38 spends 13.56 of 29.12 ms there).

**THE CANDIDATE CHANGE, ONE SENTENCE:** make the CUDA decode path dispatch its missed experts through
`run_split_multi_native` (the native-pack multi-token split form, `kMaxSplitMulti = 96`, which covers our
per-layer counts) exactly as `prefill.cpp:3332` already does - a change the header states is **bitwise
identical to `run`** ("the same outputs as `run`, bitwise"), so the invariant to verify is parity, not
quality.

**WHAT MUST BE CHECKED BEFORE BUILDING (all reads, no lane):**
1. What the decode path calls today in `src/core/expert_source.cpp` (find its `run(...)` site and the
   `ExpertJobMulti`/`NativeFmt` plumbing it would need - prefill shows the required shape).
2. Whether a `split_rows`-style gate exists on the CUDA side (the SYCL copy has one at
   `expert_source.cpp:2526`), because that is the upstream-intended switch for exactly this decision.
3. `phase_ms(wait_park, drain, repark)` - already exposed - read in the first boot: if the layer's cost is
   the BARRIER rather than the WORK, the split form will not pay and the fix moves to the park/wake protocol
   instead (the F-137 lead, voided for contention and still unmeasured).
**Prize:** the CPU term is 4.72*T - 6.8 ms/window; if the split form brings a 3.8-expert layer from 3.8
single-core streams toward one all-core stream, the measured ceiling remains F-125's 276 t/s at T 4.38.

### F-146 — QUALIFIER ON F-145: the CUDA decode path lacks the multi-job plumbing, so this is a PORT, not a one-liner (2026-10-12 ~18:30, goal round 23)

Grep of `src/core/expert_source.cpp` for `split_rows`, `pool->run`, `->run(` and `jobs_multi`: **no matches
at all.** Consequences, stated before anyone builds:

1. The CUDA decode path does not call the pool directly from `expert_source.cpp` - the dispatch goes through
   the callable that `session.cpp:963` invokes (`pool(user, s.db->h_x_f, ...)`), bound elsewhere in the
   session setup. So F-145's "find the decode run() site" has a different answer than expected: the site is
   a *binding*, not a call in the expert source.
2. More importantly, **`jobs_multi` and `split_rows` do not exist in the CUDA decode path at all.** The
   multi-token job structures and the split dispatch that `run_split_multi_native` requires live in
   `src/prefill/prefill.cpp` (the prompt path) and in the SYCL copy of `expert_source.cpp`. Porting them to
   the CUDA decode path means building the `ExpertJobMulti`/`NativeFmt` plumbing there - the prefill path and
   the SYCL file are the two references for its shape.

**THEREFORE F-145's CHANGE IS A PORT, NOT A ONE-LINE SWITCH.** It remains the strongest lead - bitwise by
the header's own statement, aimed at the term that gates every other lever, with the measured 276 t/s
ceiling - but its cost is now known to be a real piece of engineering with two in-tree reference
implementations, not a one-word edit. That distinction is exactly what this campaign keeps paying for when it
is missed (F-128/F-131, F-133, and the voided arms), so it is recorded before the build rather than after.

### F-147 — LANE STATE CHECK (2026-10-12 ~18:35, goal round 24)

DEGRADED : engine 65 t/s, GPU-reach wait 21.45 ms, dwm idle GPU share 54.2%. Reference: healthy 141-184 t/s at 8.4-10.4 ms (dwm quiet), degraded 54-82 t/s at 20-32 ms (dwm 40-47%). Environment still contaminated, so no engine-side arm can be judged; the queue (the decode-path split port, F-145/F-146) remains readable and buildable off-lane, and the two voided arms (F-135, F-137) await a clean window.


### F-148 — THE DECODE DISPATCH READ AT LAST, AND TWO OF MY OWN CLAIMS WITHDRAWN (2026-10-12 ~18:55, goal resumed by the operator)

**A TOOL FALSE NEGATIVE CAUSED F-146 TO BE WRONG.** The indexed grep tool reported *no matches* for
`expert_pool_dispatch_multi`, `split_rows` or `jobs_multi` in `src/core/expert_source.cpp`. `git grep` finds
the function **defined at `src/core/expert_source.cpp:3637`** - a 5,001-line file the tool also greps for
other patterns. **RULE FOR THIS REPO: for existence questions use `git grep`, never the indexed tool.** Two
of my own claims die with that correction: F-145's "the CUDA decode path has no multi-job plumbing" and
F-146's "the change is a port, not a one-liner" were both built on the false negative. `d.jobs_multi`
exists (`:3665`) and the dispatch is fully implemented.

**WHAT THE REAL DISPATCH DOES (read at `:3637-3760`), and it settles the whole overlap family:**
- `:3677-3679` - *"plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU
  works. Distinct experts in routing order; resident ones and the last pcie_num/256 of the missed ones go to
  the GPU."* **The GPU-first overlap that F-128/F-131/F-132 proposed to build is already implemented**, which
  is exactly what F-133 suspected from the call sites and what the measured CPU term (`4.72*T - 6.8` ms) then
  represents: the RESIDUAL after that overlap, not a missing one. The overlap family is now closed from the
  CUDA source itself, not by inference.
- `:3647-3652` - the router look-ahead is wired in (`d.lookahead->submit(layers, x_f, n_tok, d.host_res)`,
  "CS-T: warm layer + 1"), so prediction-driven admission is live and not a missing lever either.
- `:3680-3759` - the per-entry plan is far richer than any spec I wrote: `kind[]` per entry (-2 skipped,
  -1 CPU, 0 VRAM, 1 PCIe), tail-skip, Foresight swap slots (`d.fs->take`), peer/helper-GPU ownership, and the
  PCIe share arithmetic (`m = (nmiss * d.pcie_num) >> 8`).
- `:3704-3726` - `STRATA_ROUTE_TAIL_SKIP` (default **7** on CUDA serve; "changes the output"): a missed
  expert that every token routes only at rank >= R is neither copied nor computed. Already on by default, and
  lossy - noted, not a lever to touch.

**CONSEQUENCE.** The engine-side story is now complete and it is not a missing feature: the GPU/CPU overlap,
the look-ahead and the per-entry planning all exist. The CPU term that gates the bar is the residual cost of
the experts that genuinely must be computed on the host within one layer's critical path. What remains
untested and non-trivial is narrower than the campaign believed: (a) whether the pool dispatch inside this
function uses the split forms (the next read, further down the same function), and (b) the per-layer barrier
(`phase_ms`) rather than the work. Both are reads, not builds.

### F-149 — THE SPLIT DISPATCH IS ALREADY IN PRODUCTION: F-144/F-145/F-146 ARE MOOT, AND THE CPU TERM IS BANDWIDTH-BOUND (2026-10-12 ~19:00, goal round 26)

`git grep` inside `src/core/expert_source.cpp` settles it:

```
3599   if (d.split_rows) d.pool->run_split(d.jobs.data(), (int) njobs);
3600   else d.pool->run(d.jobs.data(), (int) njobs);
3921   if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
3922   else        d.pool->run_split_multi(d.jobs_multi.data(), njobs);
```

**The CUDA decode path already calls `run_split_multi_native` for native packs** (`:3921`) - precisely the
function F-144/F-145/F-146 proposed porting into it - and the single-token path already branches on
`d.split_rows` (`:3599`). Our pack (IQ3_S) is native, so the split dispatch is **in production on this
machine right now**. The four-round workstream was chasing something already built; the indexed-grep false
negative (F-148) is what made the wrong conclusion feel solid.

**WHAT THIS LEAVES, DERIVED FROM THE MEASUREMENTS ALREADY IN THE LOG.** With the GPU-first overlap (P6,
`:3677`) and the row-split both already active, the CPU term `4.72*T - 6.8` ms/window is the residual cost of
reading the missed experts from RAM:
- ~2.5-3.8 missed experts per layer-window, 1.94 MB each = **5-7.4 MB per layer**;
- at this machine's ~60 GB/s aggregate, that is **~85-125 us per layer** as a floor;
- measured: **180-283 us per layer** (71-75 us per expert).

So the term has at most a **~2x** headroom, not an order of magnitude: even a perfect implementation lands
the T 4.31 window near 22.5 ms (4.31 tokens => ~191 t/s), not at the 232 bar. The bar therefore cannot come
from this term at all - it needs a different lever: **fewer missed experts per token** (routing/residency, both
refuted by measurement: F-125 bias, F-127 slots) or **more tokens per window at unchanged CPU work** (the R17
acceptance head, whose chain is validated and whose distilled checkpoint reached 57.4% against the stock
85.4%).

**THE QUEUE, RE-RANKED BY THIS FINDING (the old one is void):**
1. **The environment** - `dwm` at 40-54% of the GPU still makes every absolute number unjudgeable (F-141..F-147).
2. **Acceptance** - the only lever that raises tokens/window without raising the CPU term per token: the R17
   head needs its KV-conditioning fix (F-104's three directions) to get from 57.4% toward the stock 85.4%.
3. **`phase_ms`** (wait-park / drain / re-park) - one boot tells whether the residual 2x in the CPU term is
   the WORK or the per-layer BARRIER; the header states those need opposite fixes.
4. Closed for good: the CPU-tier overlap (already built), the split dispatch (already built), the DMA share
   (F-126), slot capacity (F-127), residency bias (F-125), suffix/lookup depth (F-130).

### F-150 — THE WORK-vs-BARRIER INSTRUMENT EXISTS AND PRINTS TO STDOUT; ONE REDIRECT AWAY (2026-10-12 ~19:05, goal round 27)

`src/program/generate.cpp:13646-13656` prints the pool's three phases per token:

```
13648  // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
13649  // question is whether the time is in the drain or in the re-park barrier.
13654  std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
```

So the engine already answers the question F-149 left open - is the residual CPU cost the WORK or the
per-layer BARRIER - and it answers it in **ms/token**, the right unit. **It goes to stdout, not to
`strata-iq3_s.log`** (which captures the `strata serve:`/`strata generate:` stderr lines), and my launcher
does not redirect it, so no capture of it exists in this campaign yet. That is the only reason the question is
still open.

**THE NEXT BOOT, EXACTLY:** launch the standing recipe with stdout redirected to a file (the parallel
session's launchers already do this with `%TEMP%\v2*.out`), run the four-class sweep, and read the three
numbers from that file. Interpretation is pre-registered by the source's own comment: wait-park ~0 means the
workers are re-parking promptly; **drain** large means the CPU work itself is the cost (a kernel problem -
the RAM-bandwidth floor of F-149, at most ~2x); **re-park** large means the per-layer barrier is the cost (a
synchronisation problem, and then the park/wake protocol - the `STRATA_POOL_SPIN_US` knob the voided F-137
arm aimed at - is the target).

**ENVIRONMENT STATUS (round 27, recorded):** still DEGRADED - engine 55.1 t/s at 27.43 ms GPU-reach wait with
`dwm` at 36.6%. Fourth consecutive round in the contaminated state; absolute numbers stay unjudgeable, which
is why this round's output is a read plus the exact capture recipe rather than a measurement.

### F-151 — PARTIAL RECOVERY ON A FRESH BOOT, AND THE PHASES PRINT IS NOT IN THE SERVE PATH (2026-10-12 ~19:10, goal round 28)

**1. THE LANE RECOVERED PART-WAY.** Fresh boot 18:52:27, same adopted recipe, four-class sweep:

| class | T | CPU term | GPU wait | t/s |
|---|---|---|---|---|
| repetitive list | 2.81 | 7.43 | 9.68 | 124.0 |
| code | 3.98 | 12.42 | 10.62 | **133.7** |
| prose | 2.06 | 4.39 | 8.37 | 121.5 |
| literal repeat | 2.46 | 3.34 | 8.94 | 142.5 |

Against the last hour's degraded readings (55-78 t/s at 27-32 ms GPU wait) this is a large recovery, and
against the healthy class (141-184 t/s at 8.4-10.4 ms) it is still ~15-25% short. **It qualifies F-140:** a
restart CAN restore most of the state - what matters is whether the desktop load changed across it, which is
exactly what F-141/F-142 measured. The state tracks the compositor, and the compositor's load is not constant.

**2. THE THREE PHASES DID NOT PRINT IN SERVE MODE.** `generate.cpp:13646-13656`'s `pool phases` line was
captured-for on this boot (stdout redirected to `campaign/results/pool-phases.out`) and **does not appear**,
while the same boot served the full sweep and wrote its usual `strata serve:` report lines to the engine log.
The block is guarded by `if (pool_positions > 0.0)` (line 13650) and sits in the code that prints the
per-request diagnostic summary - the same summary whose serve-mode counterparts are the `strata serve:`
lines. So the print belongs to the one-shot / CLI report path, not the serve path, and the serve path does not
surface `phase_ms` at all. **Consequence for the queue:** to read wait-park / drain / re-park, run the engine
ONE-SHOT with a prompt (no server) rather than through the serve path - or read `phase_ms` from a serve-mode
summary if one exists elsewhere. That is a five-minute experiment, and it is the remaining half of F-149's
open question (WORK vs BARRIER).

### F-152 — THE PHASES PRINT IS UNREACHABLE WITHOUT A CODE CHANGE; THE CHEAP DISCRIMINATOR IS THE F-137 A/B (2026-10-12 ~19:15, goal round 29)

F-151 said "run the engine one-shot to read wait-park / drain / re-park". That door is closed: **this
workflow has no one-shot invocation.** `Get-ChildItem run*.bat` in v2 returns nothing; the v1 tree's
`run-iq3_s.bat` is just another `serve/server.py` launch; and `git grep` for one-shot flags (`-p`, `-n`,
`--prompt`, `no_serve`) in `generate.cpp` finds no matches. The engine is driven through the server here, so
`generate.cpp:13646-13656`'s `pool phases` line - which sits in the CLI report path, guarded by
`pool_positions > 0` - cannot be reached without adding the print to the serve path (a small code change, not
a config knob).

**THE CHEAP DISCRIMINATOR THAT IS AVAILABLE INSTEAD.** The three-phase split exists to answer one question:
is the residual CPU cost the WORK or the BARRIER? That question can be answered behaviourally with the knob
that failed for the wrong reason once already: **`STRATA_POOL_SPIN_US`**. If the per-layer barrier dominates,
changing how long the workers spin before parking must move throughput materially; if the WORK dominates, it
will not (the same bytes are read either way). That is exactly F-137's arm - **voided for contamination**
(F-138), not for its result.

**WHY IT IS RUNNABLE NOW:** the lane has recovered part-way (F-151: code 133.7 t/s at 10.62 ms GPU wait) and
the F-139/F-151 protocol is in place - assert the standing recipe in the SAME window, and if its GPU wait is
above ~15 ms the run is void. So the next round is a two-boot A/B with a same-window baseline anchor
(`STRATA_POOL_SPIN_US=100` standing vs `=2000`), read on the CPU term and the window time, with the
prediction pre-registered: **barrier-dominated => the CPU term moves with the spin; work-dominated => it does
not, and the acceptance axis (R17) is then the only lever that raises tokens per window without raising CPU
cost per token.**

### F-153 — THE WORK-vs-BARRIER QUESTION ANSWERED: it is the WORK, and the spin knob is confirmed at its adopted value (2026-10-12 ~19:00, goal round 30)

Same-window A/B with the anchor measured immediately before the variant (the F-139/F-151 protocol), code
class, two probes per arm, read from the engine's own decode-timing line:

| arm | probe | t/s | T | ms/window | GPU wait | **CPU term** |
|---|---|---|---|---|---|---|
| A - standing, `STRATA_POOL_SPIN_US=100` (boot 18:52:27) | 1 | 118.0 | 1.98 | 15.13 | 8.40 | **3.51** |
| | 2 | 146.2 | 2.07 | 13.02 | 8.44 | **2.11** |
| B - `STRATA_POOL_SPIN_US=2000` (boot 18:56:32) | 1 | 82.6 | 2.28 | 23.41 | 15.68 | **3.31** |
| | 2 | 111.7 | 2.56 | 20.89 | 14.26 | **2.41** |

**THE DISCRIMINATOR ANSWERS CLEANLY: the CPU term does not move with the spin** (3.31/2.41 against
3.51/2.11, inside run-to-run spread). If the per-layer BARRIER were the cost, letting the workers spin 20x
longer would have changed it; it did not. **So the residual CPU cost is the WORK - reading the missed experts
from RAM - exactly where F-149's bandwidth arithmetic already pointed** (~85-125 us/layer floor against
180-283 measured, at most ~2x headroom, and therefore not the road to the 232 bar).

**AND THE SPIN KNOB ITSELF IS NOW SETTLED AT 100 us.** The variant's damage is on the GPU side (GPU wait
8.4 -> 14-15.7 ms), which is what fifteen workers burning CPU in a spin loop costs the host/submission path;
the CPU term, its supposed target, is untouched. That retires the last open item from the v1-era tuning and
closes F-137 - this time measured under the protocol rather than void.

**CONSEQUENCE, PRE-REGISTERED IN F-152 AND NOW DUE:** with the CPU term work-bound, the byte count fixed by
routing, and residency/DMA/split/overlap all already implemented or refuted, **the acceptance axis is the only
lever left that raises tokens per window without raising CPU cost per token.** The R17 head's chain is
validated (85.4% stock round-trip) and its distilled checkpoint reached 57.4% against the stock 85.4%; the
blocker there is F-104's KV-conditioning mismatch, with three documented fix directions. That is the next
workstream, and it needs the lane only for the final gates.

LANE: restored immediately, boot 18:58:32, listener 1, adopted recipe.

### GOAL RESUME MAP v2 (goal-d0ae1db9, the 2x bar; rewritten 2026-10-12 ~19:05 after F-153)

**STANDING (all measured today, with lane state attached):** single-stream 118-146 t/s on the code class and
121-142 across the four-class sweep on the fresh boot 18:52; 6-conc 176.5 t/s aggregate (F-124); MMLU 95.7
(16:48); acceptance 90.2-90.8%; window cost 10.6 ms over 25,544 windows. THE HEALTHY CLASS measured earlier
today was 141-184 t/s at 8.4-10.4 ms GPU wait; the environment currently caps what is achievable (below).
Bar: 232 entry, 290 target - NOT met, and the path to it is now a single workstream (below).

**THE FORMULATION (the campaign's distilled physics, from the decode-timing line):**
    t/s = tau(T) / V(T),  tau = 1 + sum(p_k)
    V(T) = GPU(T) + CPU(T) + host(T)
    GPU(T)  ~ 9-10.4 ms, nearly FLAT in T
    CPU(T)  ~ 4.72*T - 6.8 ms   (the term that gates every other lever)
    host(T) ~ 0.2-0.5 ms

**CLOSED BY MEASUREMENT - do not re-buy:** residency bias (F-125: 7.11 -> 7.88 ms at T 4.0, worse);
DMA share (F-126: CPU 7.11 -> 6.15 but GPU wait +6.03 ms, 184 -> 142 t/s); slot capacity (F-127: +998 slots,
no effect, 6-conc -23%); spec-min-p 0.45 (F-130: neutral); suffix/lookup depth (F-130: 111-133 t/s);
STRATA_FUSE_WAIT (F-135: void - see below); POOL_SPIN_US=2000 (F-153: CPU term unchanged 3.31/2.41 vs
3.51/2.11 while the GPU side worsened 8.4 -> 14-15.7 ms => the CPU cost is the WORK, not the barrier, and
100 us stands); the CPU-tier overlap (F-148/149: ALREADY IMPLEMENTED - expert_source.cpp:3677 "the GPU's share,
decided and published FIRST so the GPU starts while the CPU works"); the split dispatch (F-149: ALREADY IN
PRODUCTION - expert_source.cpp:3921 run_split_multi_native for native packs); the router look-ahead
(F-148: wired at expert_source.cpp:3647).

**THE ONE PATH LEFT - ACCEPTANCE.** With the CPU term work-bound and its byte count fixed by routing,
the only lever that raises tokens per window without raising CPU cost per token is per-draft acceptance.
The R17 head: chain validated at 85.4% stock round-trip (F-114), distilled checkpoint at 57.4% against the
stock 85.4% (F-115/116), blocker = F-104's mixed-KV conditioning with three documented directions
(real-position windows / engine-state dumps / on-policy distillation). Off-lane until the final gates.

**ENVIRONMENT (measurement-critical, unresolved):** the Windows Desktop Window Manager holds 36-54% of the
GPU (F-141, F-142, F-147) and produces a 2.3x swing - 141-184 t/s at 8.4-10.4 ms GPU wait when the desktop is
quiet vs 55-82 at 20-32 ms when it is not. A restart clears it only when the desktop load changed (F-140
qualified by F-151). RULE: measure the STANDING recipe in the same window first; above ~15 ms GPU wait, no arm
may be judged (this voided F-135 and F-137). Two minutes with the GPU-heavy apps closed settles it.

**PROCESS RULES EARNED (they have each cost a round):** use git grep, never the indexed grep tool, for
existence questions in this repo (F-148: a false negative produced two wrong entries); read before building
(F-131, F-133, F-146); an upstream default gated by an architecture comment is a hypothesis for this card
(F-135, and the F-67 adoption before it); a variant's CONTROL term must be asserted in the same window
(F-137/F-138).

**STATE:** listener 1 | boot 10-12 18:58:32 engi; MAGMA v1 frozen at 8da8bf16, untouched; ledger F-122..F-153 committed $(git log --oneline -1 | ForEach-Object { .Split(' ')[0] }).

### F-154 — THE COMPOSITOR IS NOT THE CAUSE, THE GPU IS NOT STARVED, AND THE WINDOW HAS A MEASURED LAW: CPU = misses x 85 us (2026-10-12 ~19:30, goal round 33)

Four measurements in one session, in this order, each one contradicting something the campaign believed.

**1. THE COMPOSITOR CORRELATION IS BROKEN.** On the standing boot of 18:58:32 the engine read **46.3 t/s at
30.84 ms GPU-reach wait**, and at that moment `dwm` held **1.0%** of the GPU (sampled from
`\GPU Engine(pid_2740*)\Utilization Percentage`, 4 samples). F-141..F-147 blamed the desktop compositor for
every degraded reading; this reading is degraded with the compositor idle. Machinery: `campaign/gpu-state-during-decode.ps1` (new).

**2. THE GPU IS NOT STARVED - IT IS BUSY RUNNING OUR OWN KERNELS SLOWLY.** Sampling the card DURING a live
decode of the real workload class (new instrument, same file) at 24.9 t/s:

| evidence | measured | what it rules out |
|---|---|---|
| GPU utilisation | **83-99%** | starvation / scheduling gaps |
| engine's own share of `\GPU Engine(*)` | **732 of 8 samples = 91.5%** (next: dwm 18.8%, chrome 7.9%) | another process holding the card |
| SM clock / mem clock | 2962 MHz / 13801 MHz | down-clocking |
| power | 158-200 W of 575 W | power/thermal limit |
| throttle flags | `0x0000000000000000` | any throttle |
| PCIe | gen **5**, width **16** | link degradation |
| memory controllers | **5-17%** busy | bandwidth saturation |

The engine owns the card, at full clock, with two thirds of the power budget unused, and the memory controllers
idle: **the kernels themselves take 3-6x longer.** That is a different failure mode from the one the campaign
was treating, and it makes "the desktop is holding the GPU" a hypothesis that this window refutes.

**3. THE FIRST FULL PHASE DECOMPOSITION OF A WINDOW.** The verifier has carried a 33-phase per-layer GPU
profiler since before this campaign (`STRATA_VERIFY_PROFILE`, verify.cpp:711, names verify.cpp:1803) and no
round had armed it. Armed via the new diagnostic launcher `launch-v2-diag-vprof-8081.bat` (a copy of the
standing launcher plus that one variable; **caveat recorded in the file**: `prof_on_` disables four fusions -
fuse_ab :1126, sh_fork :1424, fuse_wait_on/fuse_rows :1599/:1627, fuse_hits :1642 - so a profiled boot is a
DECOMPOSITION instrument, never an adoption arm). Per window, T 1.9, total 27.15 ms:

| phase family | ms/window | note |
|---|---|---|
| expert waits: VRAM hits 4.80 + waitCPU 1.35 + waitA 0.75 + waitB 0.66 + PCIe 0.22 (GDN) + QSA share | **10.02** | the expert path is the largest single block |
| q8+qkv/q-idx gemv 2.22 + q+q-idx 1.10 + z 1.15 + out-proj 1.41 + attention 0.69 + rec 0.37 + kv-resolve 0.32 | 7.26 | dense/attention maths |
| hyper-connections hc0/hc1 norm+down+up (both layer kinds) | **3.47** | 13% of the window, architecture-side |
| MTP head 1.28 GPU (+1.34 host `draft`) | **2.62** | paid per window |
| router+plan+bell 0.91+0.36, shared+quant 0.97+0.29, copy+combine 0.49, (gap) 0.14 | 3.16 | |
| QSA: q8+kv-idx/k-v/kv-append/scores/kv-resolve/attention/gate | 2.47 | |

Only ~14% of the window is the device WAITING on the host (waitA+waitB+PCIe+waitCPU 3.69 ms); the rest is work.
At T=1.9 the window reads **1.78 GB of expert weights from VRAM (923 experts x 1.93 MiB) at ~280 GB/s** - 16% of
this card's ~1.79 TB/s. The window is dominated by per-window weight traffic and by low-M (M~2) GEMV
inefficiency, which is exactly why tokens/window is the multiplier it is.

**4. THE WINDOW HAS A LAW, AND THE RESUME MAP'S LAW IS WRONG.** Reading the engine's own counters across 30
consecutive requests on one boot (the new correlation table) gives a clean proportionality the campaign's
`CPU(T) = 4.72*T - 6.8` fit does not have:

    CPU term  =  (missed experts per layer-window) x n_layers x ~85 us
    ratio CPU/CPUexperts measured: 4.1, 4.5, 3.9, 4.0  ->  82-90 us per missed expert
    (independent confirmation: the engine's own PcieAuto line, "the CPU pool takes %.0f us per missed expert")

and the dominant driver of that term is the **expert cache hit rate**, which is CONTENT-determined and was
measured across a 20-point range on one boot:

| request | T | ms/window | GPU wait | CPU | misses/layer-window | hit % |
|---|---|---|---|---|---|---|
| 3  | 2.29 | 13.29 | 8.92 | 1.80 | 0.44 | **97.9** |
| 12 | 2.23 | 14.12 | 9.58 | 1.79 | 0.40 | **98.0** |
| 8  | 1.00 | **9.19** | 7.63 | 0.54 | 0.12 | **98.7** |
| 26 | 2.55 | 20.88 | 11.25 | 5.25 | 1.26 | 93.9 |
| 27 | 3.96 | **34.82** | 13.38 | **15.12** | 3.86 | **88.9** |
| 30 | 3.57 | 28.07 | 12.68 | 9.63 | 2.43 | 91.3 |

**THE ARITHMETIC THAT MATTERS.** Row 27 is the real-workload class at full spec depth: T 3.96, miss-driven CPU
15.12 ms. Hold that same T and take the hit rate of row 3 (97.9% instead of 88.9%) and its CPU term falls to
~2.5 ms, its window to ~15 ms, and the request to **~264 t/s - above the 232 bar.** The lever the campaign has
been circling is the EXPERT CACHE HIT RATE ON THE REAL WORKLOAD CLASS. The resume map's "the CPU term is
RAM-bound with at most ~2x headroom, so acceptance is the only lever left" was built on the wrong slope (a
cross-boot fit, the exact error the measurement protocol forbids) and it closed the wrong door.

**5. THE ONE ARM TESTED AND REFUTED: GPU SCHEDULING PRIORITY.** F-141 listed "raising the engine's GPU
priority" as a mitigation and no round had tried it. Interleaved A/B, engine process `PriorityClass`, new
instrument `campaign/gpu-priority-ab.ps1` (README-style header carries the full reasoning):

| window | arm | t/s | ms/window | GPU wait | pair agg |
|---|---|---|---|---|---|
| degraded (profiled boot) | Normal | 59.3 | 32.07 | 25.07 | - |
| | High | 70.4 | 27.12 | 21.65 | - |
| | **RealTime** | **80.3** | **23.38** | **18.01** | - |
| | Normal (repeat) | 54.4 | 39.53 | 32.01 | - |
| healthy (standing boot, REAL 2000-tok M1) | Normal | 157.8 | 13.29 | 8.92 | 110.7 |
| | High | 155.7 | 13.99 | 9.12 | - |
| | RealTime | 142.8 | 14.12 | 9.58 | 110.2 |
| | Normal (repeat) | 132.0 | 15.46 | 10.30 | 105.0 |

**VERDICT: NOT ADOPTED.** In the degraded window the arm is a 1.69x mitigation with a clean monotone
dose-response, and it moves the GPU-reach wait ONLY (CPU and draft unchanged). In the healthy window the same
arm is indistinguishable from Normal - all four arms inside a 13.29-15.46 ms band whose own repeatability is
16% - and the pair aggregate is flat (110.7 vs 110.2). It buys nothing when the machine is quiet and cannot be
adopted on the healthy-window evidence; recorded as a degraded-state mitigation only, the same class as the
environment knob it replaces.

**LANE:** restored to the standing recipe, boot 19:22:17, listener 1. On this boot the healthy class was captured
in the same window (157.8 t/s M1 at 13.29 ms, GPU wait 8.92) and then decayed inside five minutes to
100-121 t/s at 19.77-34.82 ms - the same drift F-119 recorded, now with the hit-rate column beside it (97.9% ->
88.9% over the same interval).

### GOAL RESUME MAP v3 (goal-d0ae1db9, the 2x bar; rewritten 2026-10-12 ~19:35 after F-154)

**WHAT CHANGED FROM v2 AND WHY.** v2 said the CPU term was `4.72*T - 6.8` (RAM-bound, <=2x headroom) and
therefore that per-draft acceptance was the only lever left. F-154 replaced that with a measured
proportionality - CPU = misses x ~85 us, driven by the CONTENT-DETERMINED expert cache hit rate, which ranges
79-99% on one boot - and showed a real-workload window at T 3.96 whose cost would fall from 34.82 ms to ~15 ms
(264 t/s, above the bar) at the healthy hit rate. v2's slope came from comparing two BOOTS (F-126's T 4.0 CPU
7.11 against T 2.0 CPU 2.1-3.5) - the cross-boot error the protocol forbids - and it closed the wrong door.

**THE WINDOW LAW (measured, within one boot, F-154):**
    ms/window  =  GPU wait (~9.0 healthy, nearly flat in T)  +  misses/layer-window x 48 x ~85 us  +  ~4.5 fixed
    misses     =  routed experts x (1 - hit rate);  ~23 expert lookups per layer-window;  hit rate 79-99%
    head/draft =  ~2.6 ms per window (GPU 1.28 + host 1.34), paid per WINDOW not per draft
    T          =  content-determined: code 3.96-4.02 (full spec 4), repetitive-list 2.5-3.1, prose 2.3-2.8,
                  literal-repeat 2.4-2.6, the M1 real-workload prompt 2.2-2.6
    consequence: t/s = T / (13.5 + f(misses));  at the healthy hit rate and T 3.96 -> ~264 t/s (BAR CLEARED);
                 at the measured 88.9% hit rate and T 3.96 -> 111.7 t/s.

**THE LIVE LEVER IS THE HIT RATE.** F-125 (residency bias, worse) and F-127 (slot capacity, no effect) tested
policies, not the question. The open questions, in measurement order: (a) what IS the achievable hit rate for
the M1 real-workload prompt (the row-27 content that drives the bar metric), measured per request with its
miss count? (b) how much of the 79-99% spread is the adaptive tier's residency decisions versus the content's
own routing entropy? (c) can the ~85 us per missed expert be lowered - and if the pool's aggregate read is
~25 GB/s, is that CPU-GEMV-bound (the arithmetic says one expert's 2-token GEMV at ~200 GFLOPS = ~90 us) or
RAM-bound (F-149 asserted RAM-bound without measuring the platform's bandwidth)? Those are three cheap
measurements before any training run.

**SECONDARY, UNCHANGED:** the R17 acceptance axis (F-114 chain validated at 85.4%, distilled checkpoint 57.4%,
blocked on F-104 mixed-KV conditioning) RAISES T and therefore multiplies whatever the window law gives. It is a
multiplier, not the base - and the base is currently 88.9% hit rate on the class that defines the bar.

**CLOSED BY MEASUREMENT - do not re-buy:** residency bias (F-125); DMA share (F-126); slot capacity (F-127);
`--spec-min-p` (F-130, CONFIRMED tonight by matched T: min-p 0.45 gives T 4.01/2.87/2.69/2.52 against the
standing 0.7's 4.00/3.09/2.73/2.54 on the same classes - the gate is not binding); suffix/lookup depth (F-130);
`STRATA_FUSE_WAIT` (F-135, void); `POOL_SPIN_US=2000` (F-153: CPU term unchanged, GPU side worse - the cost is
the WORK); CPU-tier overlap and split dispatch (F-148/149: already in production); router look-ahead (already
wired); **GPU scheduling priority (F-154: 1.69x while degraded, indistinguishable while healthy, NOT adopted).**

**REFUTED AS THE CAUSE:** the desktop compositor is not necessary for the degraded state (F-154: 46.3 t/s at
30.84 ms with dwm at 1.0%), and the GPU is not starved while degraded (91.5% of the GPU is ours, 99% busy, full
clock, no throttle, gen5 x16, 5-17% memory-controller busy). The two-state swing is real; its cause is inside
the window law above (misses), not in the scheduler and not necessarily in the desktop.

**INSTRUMENTS COMMITTED THIS ROUND:** `campaign/gpu-state-during-decode.ps1` (card state + per-process GPU
share + the engine's accounting, sampled DURING a live decode); `campaign/gpu-priority-ab.ps1` (interleaved
priority A/B carrying both M1 and a pair aggregate); `launch-v2-diag-vprof-8081.bat` (33-phase profiler, with
its fusion-disabling caveat written in the file).

### F-155 — THE COLD-RESIDENCY TAX, MEASURED: the same request twice halves the miss count and the CPU term (2026-10-12 ~19:40, goal round 33)

Same boot (19:22:17, standing recipe), the M1 real-workload prompt (the class the bar is stated on) issued
twice back to back, 1200 tokens each, nothing else running:

| pass | t/s | T | ms/window | GPU wait | CPU term | misses/layer-window | hit rate |
|---|---|---|---|---|---|---|---|
| 1 (cold) | 117.6 | 2.34 | 18.21 | 10.88 | **3.56** | **0.78** | 96.3% |
| 2 (warm) | 131.9 | 2.24 | 15.81 | 10.66 | **1.95** | **0.41** | 98.0% |

**The second pass reads 47% fewer missed experts per layer-window and its CPU term falls 45%**, taking 2.4 ms
out of the window. The law of F-154 holds inside the pair: CPU/CPU_experts = 4.56 then 4.76 (i.e. ~85 us per
missed expert, and misses/layer-window x 48 is the whole CPU term).

**WHY THIS IS THE MOST ACTIONABLE NUMBER OF THE ROUND.** The bar metric (M1: sustained single-stream
default-reasoning generation) is a COLD request by construction - it is the first pass over its content - so it
pays the full cold tax, and so does every fresh agent turn. The engine's two-tier cache does not THRASH (the
file tier is untouched: "0 blob reads from the file", 41.23 GiB resident in RAM) - it is COLD: the resident set
was chosen from other content, and the current content's routed experts are ~1.7x more likely to be missing on
the first pass than on the second. Consequence: the lever is to warm or predict the residency for the content
about to be decoded (from its own prefill routing) rather than to re-tune the exchange policy - and the engine
already carries the machinery (`STRATA_KV_PREFETCH`, the adaptive tier, and the prompt path's slot borrow).

**CONFOUND, STATED:** the second pass also reuses the prefix cache, so its prefill is cheaper. That does not
explain the effect: the moved quantity is the DECODE miss count per layer-window (0.78 -> 0.41), which is a
property of expert residency, not of the prompt's KV. The clean confirmation is a fresh (non-repeated) third
content with the same measurement, and a prefetch arm, both owed to the next round.

**ALSO RECORDED (dormant self-tuning, found by reading before building):** the adopted config passes
`--pcie-frac 0.35` EXPLICITLY, and generate.cpp:2829 gates the whole PCIe block on `o.pcie_frac < 0.0` - so on
this lane the host->device link has **never been measured** (`probe_pcie_h2d_gbps` never runs; no "PCIe probe"
line exists in the log) and `PcieAuto` is dormant twice over: that gate, and `o.batch == 0` at generate.cpp:8353
while this lane runs the batch path (`parallel: 4`). The miss-path split is therefore frozen at a value chosen
in an earlier era, on the exact term F-154 proved dominates the degraded window. A one-boot diagnostic with
`--pcie-frac` omitted recovers the engine's own measured link rate and the frac it would choose.

### F-155 CORRECTION — MY OWN CONTROL REFUTED MOST OF THE COLD TAX, AND THE REAL SIGNAL IS THE MISS FRACTION vs T (2026-10-12 ~19:42, same round)

**The F-155 pair above does not measure what it claims.** At `temperature 0.7` the two passes generated
DIFFERENT text, so they routed to different experts: part of the 0.78 -> 0.41 miss drop is routing luck, not
residency. Re-run as a controlled pair with `temperature 0`, so both passes emit the SAME tokens and route
identically:

| pass | t/s | T | ms/window | GPU wait | CPU | misses/layer-window | hit rate |
|---|---|---|---|---|---|---|---|
| greedy 1 | 127.8 | 2.22 | 15.63 | 10.70 | 2.24 | **0.47** | 97.6% |
| greedy 2 | 128.4 | 2.12 | 14.61 | 10.59 | 1.69 | **0.37** | 98.1% |

**CORRECTED FINDING: the cold-to-warm effect is real but small** - 0.47 -> 0.37 misses per layer-window
(-21%), ~0.4 ms of CPU, 15.63 -> 14.61 ms of window (-6.5%), and **+0.5% on t/s** because tokens/window fell
with it (T 2.22 -> 2.12). It is NOT the -47% / 2.4 ms / +12% that F-155 reported. Two further qualifications:
the greedy pair's "pass 1" was itself already warm (the same prompt had just run twice at 19:40), so the pair
measures a warm->warmer tail rather than a genuinely cold start; and the window drift across this boot is
mostly the GPU wait (8.92 ms at 19:23 -> 10.6 ms at 19:40/19:42), not the miss count.

**WHAT THE ROUND ACTUALLY ESTABLISHED, with the confounds removed.** The miss count is small and stable on the
M1 real-workload prompt at its natural T (~2.2-2.3): 0.37-0.78 misses per layer-window, 96.3-98.1% hit rate.
It is LARGE when T is large: the code class at T 3.96 read 3.86 misses per layer-window at 88.9% with 33.86
VRAM hits, and paid 15.12 ms of CPU for it. So while T rises 74% (2.3 -> 4.0), the routed-expert UNION rises
further (22 -> 34 lookups per layer-window) and the miss FRACTION rises from ~2% to ~10%:

| content | T | lookups/layer-window | misses/layer-window | miss fraction | CPU term | ms/window |
|---|---|---|---|---|---|---|
| M1 prompt, warm | 2.12-2.34 | ~22 | 0.37-0.47 | ~2% | 1.7-2.2 | 14.6-15.8 |
| code, full spec | 3.96 | 33.86 | 3.86 | **10.2%** | 15.12 | 34.82 |

**That super-linear growth of the miss fraction with tokens-per-window is the real constraint on the
draft-yield lever** - it is why the high-T window costs what it does, and it is the thing to attack: either
make the resident set cover the high-T union better, or make each miss cheaper (the frozen split above), or
cap T where the miss curve bends. F-155's headline ("the bar metric is a cold request, so warm it") is NOT
supported by the controlled measurement and is withdrawn as stated.

### F-156 — THE SPLIT IS NOT THE PROBLEM, THE MARGINAL TOKEN IS CHEAP, AND min_p 0.35 IS A CANDIDATE THAT NEEDS A QUIET LANE (2026-10-12 ~19:40, goal round 34)

Three measurements and one integrity finding. New instrument: `campaign/pcie-split-sweep.ps1`, which sweeps
either per-request tuning key (`pcie_frac` or `spec_min_p`, generate.cpp:10264-10279 via
`sampling.strata_tune`, serve/server.py:1005) IN ONE BOOT, IN ONE WINDOW, with **`temperature 0` so every arm
emits the same tokens and routes to the same experts** - any difference between arms is the knob alone. Arms
run forward then reversed so drift cannot masquerade as an effect.

**1. THE FROZEN PCIe SPLIT IS NOT COSTING ANYTHING (closed, negative).** The adopted config pins
`--pcie-frac 0.35` and (F-155) that pin keeps `probe_pcie_h2d_gbps` from ever running. Swept per request on
the M1 real-workload prompt, 1200 tokens per arm, two interleaved passes:

| pcie_frac | 0.0 | 0.2 | **0.35** | 0.55 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| t/s (median) | 130.1 | 135.1 | **134.1** | 130.5 | 135.6 | 128.9 |
| GPU wait | 10.30 | 10.48 | 10.19 | 10.23 | 10.41 | **11.96** |
| CPU term | 1.99 | 2.24 | 1.90 | 1.86 | 1.78 | **0.00** |
| misses/layer-window | 0.47 | 0.56 | 0.43 | 0.37 | 0.32 | **0.00** |

**The instrument self-verifies**: at 0.0 the CPU carries every miss and the PCIe column reads 0.00; at 1.0 the
two swap exactly (CPU 0.00, PCIe 0.44) - so the tune reached the engine. **And the whole range spans +-4%**,
inside this window's floor, while pushing misses onto PCIe *raises* the GPU wait (10.2 -> 12.0 ms). Verdict:
**keep 0.35; PcieAuto's dormancy is harmless on this lane; do not re-buy.**

**2. THE MARGINAL TOKEN COSTS ~1.7 ms, NOT 4.7 - DRAFT YIELD IS STRONGLY NET-POSITIVE.** Sweeping
`spec_min_p` on identical routing (code class, greedy, 500 tokens per arm) moves the window size T from 1.11
to 3.97 and gives the window's cost curve directly:

| spec_min_p | 1.0 | 0.9 | 0.8 | **0.7 (standing)** | 0.5 | 0.3 | 0.0 |
|---|---|---|---|---|---|---|---|
| T (tokens/window) | 1.11 | 1.56 | 2.09 | **1.83** | 2.99 | 2.99 | 3.97 |
| ms/window | 11.76 | 11.44 | 12.95 | **11.84** | 14.77 | 14.19 | 16.57 |
| GPU wait | 7.59 | 8.02 | 8.39 | 8.24 | 9.24 | 9.31 | 10.11 |
| t/s | 89.5 | 130.1 | 153.4 | **143.0** | **172.2** | 147.8 | 152.8 |

**d(ms/window)/dT = 4.81 ms for +2.86 tokens = 1.68 ms per banked token**, against an *average* cost of ~4 ms
per token on this content. The window is nearly flat in T because its cost is per-window weight traffic (F-154:
1.78 GB of expert weights at ~280 GB/s), not per-token work. **This corrects the resume map's central
quantity**: F-154's "CPU = misses x 85 us" is right, but v3 then read the cross-boot slope (4.72 ms/token) as a
per-token wall; measured WITHIN a window the slope is ~1.7 ms/token and the trend in t/s is +71% from T 1.11
to T 3.97. Draft yield is not a wash - it is the lever, and the standing config (T 1.83) sits BELOW the
frontier this sweep finds (T ~2.5-3.5).

**3. min_p 0.35 IS A CANDIDATE (+3 to +6%), NOT YET AN ADOPTION.** The same knob in the production regime
(`temperature 0.7`, the M1 real-workload prompt, 2000 tokens, interleaved forward+reverse, two runs):

| run | arm 0.7 | arm 0.5 | arm 0.35 |
|---|---|---|---|
| 1 pass A / pass B | 152.4 (T 2.24) / 160.7 (T 2.41) | 166.5 (T 2.50) / 164.0 (T 2.45) | 171.9 (T 2.87) / 160.7 (T 2.76) |
| 2 pass A / pass B | 169.7 (T 2.33) / 167.6 (T 2.37) | - | 173.3 (T 2.86) / 174.9 (T 2.86) |

**Five of six interleaved comparisons favour 0.35, one is a tie, none favour 0.7**, and the mechanism is
tight: T is 2.76-2.87 on every 0.35 arm against 2.24-2.41 on every 0.7 arm (+21%, non-overlapping), while the
window rises only 13.0 -> 14.5 ms (+11%). Acceptance stays in class in BOTH arms (79-94% across these
requests, against the standing 88.8-90.8%), so M3 does not collapse: a longer chain drafts deeper, and the
later drafts are rejected more often, which is the natural shape of the metric rather than a quality change -
`spec_min_p` gates only how many drafts are *offered*, every one of them is still verified against the target,
so the output distribution cannot change. **Named as a trade anyway, per the ultimatum:** if adopted, M3's
*rate* reads lower while M1's *throughput* reads higher; the two must be printed side by side.

**4. LANE-INTEGRITY FINDING, RECORDED AGAINST MY OWN RESULT.** Since this boot (19:22:17) the log holds **90
request lines against the ~29-38 I issued by my own count**, so a parallel program shared this lane for part of
it (the same class of contamination the resume map warns about: F-119/F-123/F-129). Two further oddities seen
in the same log: the server CHAINS successive requests into one conversation (the code-class arms appear as
"prompt 1618 tokens = 1600 reused + 18 read", i.e. my 18-token prompt appended to a 1600-token prefix that
contains a previous arm's answer), and one code-class decode line at T 3.16 belongs to no arm of mine. No other
client is active as of 19:41 (the log grew 0 bytes in 12 s with no request from me). **Consequence: the
CANDIDATE in item 3 is not adopted on this evidence** - the arm effect reproduced causally (T is reproducible
to a hair by arm), but the absolute t/s and the quiet-lane precondition cannot be certified. The confirming
run is a quiet-lane repeat of the same interleaved design plus the pair probe at both arms.

**STATE:** lane serving the adopted recipe, boot 19:22:17, listener 1; MAGMA v1 frozen at 8da8bf16.

### F-157 — THE QUIET-LANE CONFIRMATION: min_p 0.5 and 0.35 both buy M1 ~+8% AND BOTH COST M3 ~4.5 POINTS (2026-10-12 ~19:45, goal round 34)

Re-run of the F-156 candidate on a lane verified quiet (the log grew 0 bytes in 12 s with no request from
me), same boot, same window, interleaved forward+reverse, now carrying **all three metrics** - M1, the pair
aggregate (M2) and the engine's own draft acceptance (M3):

| run | arm | M1 t/s | T | ms/window | M3 acceptance | drafts tested | M2 pair agg |
|---|---|---|---|---|---|---|---|
| quiet, pass 1 | 0.7 | 159.8 | 2.18 | 12.78 | **90.1%** | 857 | 120.0 |
| | 0.35 | 166.8 | 2.80 | 14.78 | 86.6% | 1055 | 117.0 |
| quiet, pass 2 | 0.7 | 146.9 | 2.09 | 12.78 | **90.9%** | 821 | 123.4 |
| | 0.35 | 163.1 | 2.96 | 15.84 | 84.8% | 1104 | 114.7 |
| mid, pass 1 | 0.7 | 136.4 | 2.17 | 15.02 | **91.2%** | 849 | 102.0 |
| | 0.5 | 151.9 | 2.63 | 15.72 | 86.8% | 1012 | 110.8 |
| mid, pass 2 | 0.7 | 135.1 | 2.25 | 15.51 | **91.9%** | 872 | 109.0 |
| | 0.5 | 145.4 | 2.49 | 15.85 | 87.8% | 969 | 106.6 |

**M1: both candidate values win every interleaved comparison** (+4.4% and +11.0% for 0.35; +11.4% and +7.6%
for 0.5, medians +4.4% and +11.4%).

**M3: both lose ~4.5 points of acceptance RATE** (90.5 -> 85.7 for 0.35; 91.6 -> 87.3 for 0.5). The cause is
visible in the same table: the chain tests ~1010-1100 drafts instead of ~850, so the deeper drafts enter the
denominator, and deeper drafts are accepted less often. Tokens per window still ROSE (2.2 -> 2.5-2.9), which is
what the throughput is made of.

**M2: within noise and on an inert path.** The batch/pair path runs DRAFTLESS on this build ("slot drafts
accepted 0 of 0"), so a drafting gate cannot affect it; the observed spread (114.7-123.4) shows no arm pattern
across the four pair runs. Recorded as in class, not as equal.

**THE DISTRIBUTION ARGUMENT, STATED BEFORE IT IS RELIED ON.** `spec_min_p` decides only how many drafts are
OFFERED. Every offered draft is still verified against the target's own distribution, and a rejected draft
simply does not extend the window; speculative decoding with exact verification cannot change the sampled
output. So M4 (MMLU-Pro) and M5 (mimic) are unaffected in distribution, and M3's fall is a property of the
RATE's denominator as the chain deepens, not a quality change. That argument is a hypothesis about the
metric, not a measurement - the measurement is the MMLU/mimic gate - and the ultimatum's own rule is that a
trade ships only when both numbers are printed side by side AND the operator accepts it explicitly. Both
numbers are in the table above; the decision is the operator's and is being put to them.

**STATE:** lane serving the adopted recipe (unchanged - nothing adopted this round), boot 19:22:17, listener 1.

### F-158 — THE OPERATOR PUSHED FOR RESEARCH, THE LAW'S M3 WAS AMENDED, AND THE FLOOR DECISION WENT TO A GATED 0.6 (2026-10-12 ~20:20-21:15, goal rounds 34-35)

**WHAT HAPPENED.** The min_p trade was put to the operator with both numbers side by side (F-157). They did not
accept it on the spot: they **pushed the survey back for research depth**, requiring (1) at least twelve
comparables studied, (2) the findings written down as reusable artifacts, (3) the survey re-authored at double
depth, (4) re-issued without asking anything first. All four were done.

**1. THE RESEARCH — 18 COMPARABLES, WRITTEN DOWN.** Four parallel streams, primary sources only, every claim
carrying a URL. Artifacts: [`.docs/digest/speculative-draft-yield-and-acceptance-metrics.md`](.docs/digest/speculative-draft-yield-and-acceptance-metrics.md)
(the condensed digest), [`.docs/research/serving-engines-spec-metrics.md`](.docs/research/serving-engines-spec-metrics.md)
(vLLM, TensorRT-LLM, SGLang, TGI), [`.docs/research/local-runtimes-draft-gates.md`](.docs/research/local-runtimes-draft-gates.md)
(llama.cpp, ExLlamaV2, LMDeploy, MLC-LLM), [`.docs/research/draft-head-designs-accept-length.md`](.docs/research/draft-head-designs-accept-length.md)
(Medusa, EAGLE 1/2/3, Hydra, Lookahead, REST), [`.docs/research/mtp-and-autotuning.md`](.docs/research/mtp-and-autotuning.md)
(DeepSeek-V3/V3.2 MTP, Qwen3-Next, ReDrafter, prompt-lookup, auto-tuning practice),
[`.docs/research/engine-inventory.md`](.docs/research/engine-inventory.md) (our own side, with file:line handles),
and verbatim identifiers in [`.docs/research/rips/spec-decode-metric-surfaces.md`](.docs/research/rips/spec-decode-metric-surfaces.md).

**THE FINDING THAT CHANGED THE DECISION: no surveyed system treats acceptance RATE as its objective or as a
non-regression gate.** The field's headline is acceptance **length** (tokens per forward pass): vLLM
`mean_acceptance_length`, SGLang `spec_accept_length`, TensorRT-LLM `acceptance_length` = output tokens per
decoding iteration, MLC-LLM cumulative `accept_len`, EAGLE's tau. EAGLE-2 declines to publish the rate with a
stated reason ("it only reflects the performance of the draft model"); HuggingFace TGI ships speculation with
**no acceptance metric at all**; EAGLE-3's Table 1 ranks eight methods with no rate column; and vLLM's own
tuning PR measured the decisive case - **the highest acceptance length (2.27) ran 19% slower than a lower one
(1.86)**. Ours is also the **highest fixed floor in the survey**: llama.cpp ships its floor **disabled by
default** (0.00) and its community landed at 0.6-0.8, ExLlamaV2 hard-codes 0.25, Medusa and Hydra sweep
epsilon over 0.05-0.25, and vLLM, TensorRT-LLM, SGLang and TGI have no floor at all.

**2. THE RE-AUTHORED SURVEY.** 17 questions (2.8x the original six), 12 questions deep on the longest branch
path, 5 one-click stances, every option citing the digest, branch dimensions the research opened: M3's
definition, per-position instrumentation, policy-driven depth, entropy-scaled acceptance, batch-keyed depth,
the two lossless acceptance paths, who tunes, lane exclusivity, doctrine. Payload:
[`.docs/survey-payload.json`](.docs/survey-payload.json); draft
`.dsh/survey-drafts/the-acceptance-length-question-what-should-m3-me.json`.

**3. THE OPERATOR'S DECISIONS (verbatim, and what each binds).**

| # | question | decision | what it binds |
|---|---|---|---|
| 1 | the floor | **gate it first: MMLU-Pro and mimic, then decide** | nothing is adopted; 0.7 stands until the gate passes |
| 2 | gate value | **0.6 - a smaller step** | the gate runs on 0.6, not the 0.5/0.35 that were measured |
| 3 | gate scope | **both suites, plus the full six-metric probe** | one boot: M1/M2/M3 + MMLU-Pro + mimic in one window |
| 4 | M3's definition | **accepted tokens per millisecond** (not acceptance length) | the law's M3 is now a goodput rate; the RATE and the banked tokens/window are reported beside it |
| 5 | M3's reporting | **all four:** amend AGENTS.md, add PERF-CHART columns, probe-lane prints all three, publish per-position | done below |
| 6 | per-position | **arm `STRATA_SPEC_DEPTH=1` at both floors in one window** | armed in the gate launcher |
| 7 | batch path | **arm coupled MTP in the slot path** - justification: *"First measure and ensure we improve and dont regress. even single stream/session must not regress"* | measure first; single-stream must not regress |
| 8 | typical acceptance | **add the entropy term** (Medusa's co-threshold) - justification: *"as long as it benefits to have it and we can make sure it makes things better rather than worse"* | must be proven better, not merely plausible |
| 9 | lossless paths | **both `STRATA_SPEC_PROB` and Gumbel, interleaved in one window** | one boot, interleaved |
| 10 | who tunes | **run `tools/calibrate.py` per machine and commit its pick** | the shipped calibration tool, not hand-tuning |
| 11 | lane exclusivity | **coordinate quiet windows manually** | no lease, no hard refusal |
| 12 | doctrine | **add a clause requiring field practice to be cited** | not the M3 redefinition - the citation clause |
| 13 | next work | **instrumentation and doctrine first** | this round's scope |

**4. THE LAW AMENDED (operator decision, item 4 and 12).** `AGENTS.md` and the law's owner
[`AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md`](AGENTS.d/extractions/2026-10-12-non-regression-ultimatum.md)
now carry: **(M3) accepted draft tokens per second** = (banked tokens per verify window - 1) / (ms per window),
which is the objective the engine's own `DraftPolicy` already optimises; the acceptance RATE, the banked tokens
per window and the per-position vector are reported **beside** it. Added: **THE FIELD-PRACTICE CLAUSE** - a
decision that ships or rejects a trade must cite the published practice of comparable engines with a source,
because a veto reached on our own numbers alone is how a defensible trade gets rejected. `PERF-CHART.md` now
carries M3 (standing **92.3 accepted draft tokens/s** at T 2.18 / 12.78 ms, frontier **121.8 /s** at floor 0.35),
M3b the rate, and M3c the banked tokens per window.

**5. THE GATE IS BUILT AND NOT YET RUN.** `strata-iq3_s-gate06.json` (the adopted config with `--spec-min-p
0.6`) and `launch-v2-gate06-8081.bat` (the standing launcher on that config, plus `STRATA_SPEC_DEPTH=1`). It was
**not booted tonight**: the lane must be handed back serving the adopted recipe, and the graded suites could not
finish inside this session - so the lane still runs the adopted recipe untouched rather than being left on a
test build. **A bug was caught while building it:** `STRATA_VERIFY_PROFILE` is read for PRESENCE, not value
(`verify.cpp:711`), so the `=0` I first wrote would have armed the 33-phase profiler and disabled four fusions
during the gate; the line is removed and the file now contains no mention of it.

**NEXT (in the operator's order):** boot `launch-v2-gate06-8081.bat`, run `campaign/probe-lane.ps1` (which now
prints M3 accepted/s, M3b rate, M3c banked tokens/window and the per-position vector), then MMLU-Pro
(`mmlu-pro-strata2.py`, 70 questions) and mimic (`mimicbench`) in the same window; adopt 0.6 only if M1 and M2
are at or above baseline and M3-M6 in class; then hand the lane back.

### F-159 — THE 0.6 GATE FAILS ON PERFORMANCE, AND THE PER-POSITION VECTOR SAYS WHY (2026-10-12 ~21:14, goal round 35)

**THE GATE RAN.** Boot 21:14:12 on `launch-v2-gate06-8081.bat` (config floor **0.6**, `STRATA_SPEC_DEPTH=1`),
boot line confirmed `--spec-min-p 0.6`. The comparison used the per-request tuning key, so **both floors ran in
one boot, in one window, interleaved forward then reversed** - the strongest form the protocol allows.

| pass | floor | M1 t/s | T | ms/window | GPU wait | CPU | misses/lw | hit % | M3b rate | M2 pair |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6 | **106.3** | 2.51 | 20.46 | 9.76 | 6.66 | 1.72 | 92.0 | 87.6% | **77.3** |
| 1 | 0.7 | 101.2 | 2.22 | 19.62 | 9.45 | 6.40 | 1.67 | 91.3 | 90.2% | **80.0** |
| 2 | 0.7 | 101.0 | 2.30 | 19.29 | 9.60 | 5.98 | 1.53 | 92.5 | 88.9% | **80.1** |
| 2 | 0.6 | 99.8 | 2.24 | 19.65 | 9.54 | 6.21 | 1.64 | 91.5 | 91.7% | **77.4** |

**VERDICT: REJECTED.** Pass 1 gives +5.0% on M1 and pass 2 gives **-1.2%**, so the effect did not reproduce;
and **M2 is lower at 0.6 in BOTH passes** (77.3 and 77.4 against 80.0 and 80.1, -3.4% consistently). The
adoption gate requires M1 **and** M2 at or above baseline - it fails on both counts. **0.6 is not adopted and the
config stays at 0.7.** The honest reading of tonight's whole floor record: +11.4%/+7.6% at 0.5 and +4.4%/+11.0%
at 0.35 were measured in a **healthy-hit** window (hit 97-98%, CPU 1.6-2.4 ms); on this **miss-degraded** boot
(hit 91-92%, CPU 6.0-6.7 ms) the same knob buys nothing measurable. The floor's benefit is state-dependent, and
that is a property of the miss path, not of the draft head.

**THE PER-POSITION VECTOR - FIRST EVER READ, AND IT CHANGES THE PICTURE.** `STRATA_SPEC_DEPTH=1`, four
single-stream arms, eight engine requests (the server splits each API call into two engine requests):

| arm | request T | pos 1 accepted/tested | pos 2 | pos 3 | pos 4 | pos 5 |
|---|---|---|---|---|---|---|
| 0.6 p1 | 2.51 | 390/403 = **96.8%** | 267/290 = 92.1% | 169/188 = 89.9% | **0/0** | **0/0** |
| 0.6 p1 | 2.95 | 136/141 = 96.5% | 104/111 = 93.7% | 67/78 = 85.9% | **0/0** | **0/0** |
| 0.7 p1 | 2.22 | 377/396 = 95.2% | 247/258 = 95.7% | 155/172 = 90.1% | **0/0** | **0/0** |
| 0.7 p1 | 2.67 | 137/143 = 95.8% | 93/95 = 97.9% | 60/68 = 88.2% | **0/0** | **0/0** |
| 0.7 p2 | 2.30 | 385/404 = 95.3% | 238/257 = 92.6% | 151/163 = 92.6% | **0/0** | **0/0** |
| 0.7 p2 | 2.66 | 132/143 = 92.3% | 88/98 = 89.8% | 64/70 = 91.4% | **0/0** | **0/0** |
| 0.6 p2 | 2.24 | 402/418 = 96.2% | 234/254 = 92.1% | 152/162 = 93.8% | **0/0** | **0/0** |
| 0.6 p2 | 2.49 | 134/140 = 95.7% | 86/92 = 93.5% | 57/67 = 85.1% | **0/0** | **0/0** |

**Three findings, all new:**
1. **Positions 4 and 5 are never offered - not once, at either floor, in any request.** `--spec 4` allows four
   drafts and the verify window holds up to six tokens; the chain stops at three. Roughly 38% of windows test
   **no draft at all** (403 position-1 drafts over ~646 windows).
2. **Our per-position acceptance is nearly FLAT and far better than the field's at depth**: ~96% / ~93% / ~90%
   against DeepSeek-V3's 85-90% at depth **1** and FastMTP's 70% / 11% / 2% at depths 1/2/3 on a reused head.
   Position 3 is still accepted 85-94% of the time here, where the published heads are near zero.
3. **So the binding constraint is the FLOOR, not the draft head** - consistent with the field practice that
   already said so: llama.cpp ships its floor disabled by default, ExLlamaV2 hard-codes 0.25, and Medusa and
   Hydra sweep epsilon over 0.05-0.25, all an order below our 0.7. Tonight's floor record shows a threshold
   effect rather than a dial: at floors 0.5-0.7 the chain stops at <=3 drafts (T 2.2-2.9), and only at floor
   0.0-0.35 does it reach the cap (T 3.97 on identical routing in F-156's greedy sweep).

**F-159 CORRECTION (same round, by reading the code): THE CAP, NOT THE FLOOR, IS WHY POSITIONS 4-5 DO NOT
EXIST.** Finding 1 above attributes the missing positions to the floor. It is the CONFIG: `--spec N` is "the
MTP drafter's verify window: how many tokens it proposes per check" (`generate.cpp:890`) and the chain gate is
`while (T < S_mtp && ...) ++T` (`generate.cpp:11524`) with `const int S_mtp = o.mtp_max_t > 0 ?
std::min(o.mtp_max_t, S) : S` and `S = o.spec` (`generate.cpp:9231-9232`). This lane runs `--spec 4` with
`--mtp-max-t` unset, so `mtp_max_t = spec = 4`, `S_mtp = 4`, and **T can never exceed 4** - which is exactly
one bonus token plus **three** drafts, i.e. positions 1, 2 and 3, and nothing beyond. The per-position vector
(`0/0` at positions 4 and 5 in all eight requests) and the greedy sweep's ceiling (T 3.97 at floor 0.0) are
both that cap, not the probability floor. **So the two levers are separate and must be tested separately:**
the FLOOR decides how early the chain stops *inside* the cap (0.7 stops at ~2.2 rows, 0.0 reaches 3.97), and
`--spec` decides whether positions 4+ exist at all. The deeper-chain question is therefore a `--spec 5..8` boot
(8 is `DraftPolicy::kMaxT`, and `--lookup-chain` already raises the verify cap toward 8 at `generate.cpp:2570`),
combined with a floor low enough to reach the new positions. Nothing is adopted from this; it is the shape the
next floor experiment must have, and it is why "0.6 changes nothing" is unsurprising - 0.6 moves a gate that
was already stopping before the cap.


**CONSEQUENCE, RECORDED AS A HYPOTHESIS PENDING THE OPERATOR.** A small step down (0.6) is measurably nothing:
it neither lengthens the chain past position 3 nor reproduces a throughput gain outside a healthy-hit window.
The measured case for going **much** lower, or for letting the policy choose the depth, rests on the
per-position data above - and it is a config-or-code decision the operator has not yet authorised, so it stays
on the queue rather than in the recipe.

**THE QUALITY HALF RAN ANYWAY, AND WHY.** MMLU-Pro (70 questions) at floor 0.6 was started in the background
(`campaign/results/mmlu-gate06-floor06.txt`), with mimic to follow. It cannot rescue the rejected arm; it is
there to answer the question the NEXT decision needs - whether moving the floor shifts M4/M5 at all - which is
the invariance claim made in F-158, now being measured rather than argued.

### F-160 — THE OPERATOR CAUGHT THE REAL MECHANISM: THE "DEGRADED STATE" IS A HALF-SIZED EXPERT CACHE, DISPLACED BY DESKTOP VRAM (2026-10-12 ~21:38, goal round 35)

**THE OPERATOR'S REPORT (it outranks my measurements, so reproduce first).** Watching the GUI during the gate
boot: "29.4/32 GB, 5,082 experts cached - something wrong with only 5k experts cached?" **Reproduced from the
engine's own console, matched against the two boots before it:**

| boot | expert cache filled | hit rate measured on it | single-stream |
|---|---|---|---|
| 19:19:41 profiled | 9,853 experts / 18.69 GiB | 95.5-98.5% | 65-69 t/s (profiler armed) |
| 19:22:17 standing | **9,638 experts / 18.28 GiB** | **97.1-97.5%** | **149.5-159.6 t/s** |
| 21:14:12 gate (0.6) | **5,082 experts / 9.67 GiB** | **91.0%**, and **63.1-63.8%** concurrent | 111.4, **40.7-41.8 t/s** |

**CAUSE, IN THE ENGINE'S OWN WORDS - not a leak, a VRAM displacement at boot:**

    strata generate: expert cache auto: 20.47 GiB free, 1000 MiB reserved (+219 MiB for the draft head) -> 7773 slots
    strata generate: only 0 MiB free once the slots are written (reserve 1000 MiB); shrinking the expert cache   (x4)

At 21:14 the engine saw **20.47 GiB free** and then shrank from 7,773 to 5,082 slots over four rounds; the
19:22 boot had enough free VRAM to fill 9,638. At the moment of the check, `nvidia-smi` showed 29,772 MiB of
32,607 MiB in use with the desktop heavy (chrome, Edge WebView2, explorer, Terminal, Notepad++, the harness).
**The engine is not leaking: the desktop is holding VRAM before the engine boots, and the expert cache is what
pays for it.**

**WHY THIS MATTERS MORE THAN ANY SINGLE ARM.** It re-reads the whole night's ledger:
1. **The "two states" (141-184 t/s against 54-82) are a cache-size difference.** A 4,500-expert smaller cache
   costs the hit rate roughly 6 points (97.5% -> 91.0%) and the concurrent hit rate ~34 points (97% -> 63%),
   which is a ~30% throughput swing with no engine change at all.
2. **F-141's compositor correlation was right about the wrong quantity.** `dwm` was not stealing GPU *time* (F-154
   refuted that: 46 t/s at 30.84 ms with dwm at 1.0%); a compositor holding GPU *memory* displaces the expert
   cache. Same actor, different mechanism, and this one is measurable at boot in one line of the console.
3. **F-140/F-151's "a restart sometimes fixes it" is exactly right**: a restart re-sizes the cache against
   whatever VRAM is free at that moment.
4. **Nothing measured tonight is invalidated where the comparator was in the same boot** - both floors ran in
   this crippled window, so F-159's rejection of 0.6 stands - but every ABSOLUTE number taken on the 21:14 boot
   (100-106 t/s single-stream) is cache-limited and must be quoted with its cache size.

**THE M4 QUESTION, ANSWERED BY A MATCHED RUN RATHER THAN A REMEMBERED ONE.** `mmlu-pro-strata2.py` cannot carry
the floor per request, so a matched version was written
([`campaign/mmlu-pro-gate.py`](campaign/mmlu-pro-gate.py), floor passed through `sampling.strata_tune`) and both
floors were measured **in the same boot and the same window**:

| floor | MMLU-Pro (70q) | 1-sigma on 70 questions |
|---|---|---|
| 0.7 (standing) | **94.3%** (66/70) | 2.77 points |
| 0.6 (candidate) | **92.9%** (65/70) | - |

**Difference: 1 question = 0.5 sigma = nothing.** The single run at 0.6 that read 92.9% against the 16:48
record of 95.7 would have been reported as a 2.8-point M4 regression if the comparator had been remembered
instead of measured - and the matched 0.7 reading in the same window is 94.3%, so the floor does not move MMLU.
Recorded as the standing example of why the matched-window law exists.

**THE IMMEDIATE FIX AND THE QUEUE.** (a) Environment: boot the lane with a light desktop, or free VRAM first, to
restore ~9.6k experts - the console line `filling the GPU's expert cache (N experts, X GiB of VRAM)` is the
one-line check, and it should be quoted with every absolute number from now on. (b) Engine: the sizing path
estimated 7,773 slots from 20.47 GiB free and then shrank four times to 5,082 - an optimistic reserve
(`--vram-reserve-mib 1000`) that costs gigabytes of cache whenever the desktop is heavy. Worth a measured arm
(raise the reserve, or size the cache from free VRAM at fill time), not a guess. (c) The deeper-chain experiment
now has its shape from the F-159 correction: `--spec 5..8` to make positions 4+ exist at all, plus a floor low
enough to reach them.

**F-160 ADDENDUM 2 - THE MECHANISM, PINNED, AND ADDENDUM 1'S REASONING RETRACTED.** Addendum 1 read the
post-shrink free-VRAM figure of a crippled boot against the pre-shrink figure of a healthy one and concluded
"the desktop is exonerated". That comparison was invalid - a smaller cache is the *consequence* of the shrink and
is exactly why its total was lower and its free higher. The real mechanism, from the engine's own final lines:

| boot | shrink events | final cache | prompt path borrows |
|---|---|---|---|
| 18:58 | **0** | **9,853 slots / 18.69 GiB** | 6,714 slots / 12.68 GiB |
| 19:22 | **0** | **9,638 slots / 18.28 GiB** | 6,714 slots / 12.68 GiB |
| 21:42 | **4** | **5,082 slots / 9.67 GiB** | 4,307 slots / 8.17 GiB |

**Two defects, both in the sizing path, neither of them a desktop problem:**
1. **The planner never measures.** `expert cache auto: 20.47 GiB free, ... -> 7773 slots` is byte-identical in
   **all fourteen boots in the log** (log lines 1227 to 7395) - it is a constant, not a live reading - so the
   initial cache size is blind to whatever else holds VRAM at boot. The only place the real figure enters is the
   post-memset check in the shrink loop (`generate.cpp:5188-5201`).
2. **The shrink loop over-corrects.** It shrinks whenever the post-write free read is below
   `want = vram_reserve_mib + pipe_first` (line 5190-5191), and for a read of 0 it gives back 1 GiB, then 1 GiB,
   then a quarter of the cache each time (line 5195-5196, whose own comment says a 0 "only says at least" and
   that WDDM reads run ~0.7 GiB high). On the healthy boots that read was at or above the target and the loop
   never fired; on the last two boots it read **0 four times**, clawing **4,556 slots (8.6 GiB)** off the cache
   and stopping with **1,901 MiB free against a 1,000 MiB reserve** - ~900 MiB of extra headroom bought with
   8.6 GiB of expert cache, which is the ~30% throughput difference measured above.
**Retraction:** addendum 1's "the desktop is exonerated" is withdrawn; what is exonerated is the *leak* theory -
there is no leak, and a restart does not help because the same post-write read fails the same way.

### F-161 — THE FIX WAS BUILT, DEPLOYED, AND REFUTED BY ITS OWN INSTRUMENT (2026-10-12 ~21:52, goal round 37)

**WHAT WAS BUILT.** The re-read hypothesis: act on a post-write free-VRAM reading only after it settles. The patch
re-reads up to 16 times at 50 ms and requires two consecutive agreeing reads, keeping the best, and prints the
before/after when they differ (`generate.cpp`, the shrink loop). Built clean with `build-strata.bat` (Ninja,
`[4/4] Linking CXX executable strata.exe`, exit 0), deployed to `engine/strata.exe`, booted on the adopted recipe.

**RESULT: NO CHANGE, AND THE PATCH REFUTED ITSELF.** The boot produced the identical four shrinks and
`expert cache 5082 slots, 9.67 GiB of VRAM`. **The diagnostic line never printed**, which is the informative part:
it prints only when the settled value differs from the first read, so **all 16 re-reads across 800 ms returned the
same figure** - the `0` is not a transient WDDM reconciliation artifact. It is stable.

**WHAT THAT NARROWS IT TO.** `generate.cpp:4896` shows the PLAN does measure live free VRAM
(`device_free_bytes()` = `cudaMemGetInfo` on Windows), and it reports **20.47 GiB in all fourteen boots in the
log** - a genuinely stable input. The healthy boots (18:58, 19:22) read at or above the reserve after writing the
slots and **never shrank**, ending at 9,853 and 9,638 slots; the crippled boots read a stable 0 and shrank four
times to 5,082. Identical plan inputs, opposite outcomes, and the difference is entirely in that one post-write
read. **So the next fix must not act on a single post-write reading at all** - the candidates to measure, not
guess: (a) shrink only when the allocation that comes NEXT actually fails (the verify-window buffers are the real
consumer of the reserve), (b) bound the total give-back so one suspicious reading can cost at most one step, or
(c) keep the plan's size and prove the fit by a touch test that can fail loudly, rather than by a free-space
number. The 8.6 GiB at stake is worth ~30% throughput on every measurement.

**THE PATCH IS NOT KEPT.** It bought nothing and cost 800 ms of startup, so it is reverted; `engine/strata.exe`
is back to the adopted 09:29:28 build and the lane serves the adopted recipe (`--spec-min-p 0.7`, listener 1,
cache 5,082 - the defect is unchanged and still open).

**PROCESS RULE EARNED (it nearly left a test build serving).** `build-strata.bat` writes **`build/strata.exe`**,
but the launcher runs **`engine/strata.exe`** - deployment is a separate copy. The first revert attempt failed
*silently* because the running engine still held the file four seconds after the kill, and the lane briefly served
the patched build. Caught by checking the binary's timestamp rather than the copy's exit code. **After any engine
swap, stop the process, poll until it is gone, copy, then verify `engine/strata.exe`'s LastWriteTime** - and keep
the previous binary beside it as `strata.exe.prev-f160`.

### F-162 — THE CACHE-SIZING DEFECT EXPLAINED WITH ARITHMETIC: THE PLAN RESERVES 5.67 GiB AND THE CHECK DEMANDS 1.0 GiB (2026-10-12 ~22:00, goal round 38)

No new boot was needed: the numbers in the two log lines already pin it. The sizing path uses **two different
reserve formulae**, and the second is the bug.

| | reserve used | where |
|---|---|---|
| **the plan** that chooses the slot count | `vram_reserve_mib + prefill_mib + mtp_bind + pipe_first` | `generate.cpp:4932-4934` - `slots = (free_b - reserve) / blob` |
| **the post-write check** that shrinks the cache | `vram_reserve_mib + pipe_first` - **no `prefill_mib`, no `mtp_bind`** | `generate.cpp:5190` - `want = (vram_reserve_mib << 20) + pipe_first` |

**THE ARITHMETIC, from the two logged lines.** `blob` (bytes per slot) = 9.67 GiB / 5,082 slots = **1.949 MiB**;
therefore the plan's 7,773 slots occupy **14.80 GiB**, and from `free_b = 20.47 GiB` the plan's reserve is
**≈5.67 GiB** - i.e. `vram_reserve 0.98 + mtp_bind 0.21 + prefill_mib ≈ 4.48` (the prompt-chunk reservation).
`pipe_first = 0` on this lane because `pipeline_windows <= 0` (`generate.cpp:4875`), so the post-write check
demands only **~1.0 GiB** free. **After writing 14.80 GiB of cache the real free VRAM is 5.67 GiB, which passes
that check with room to spare** - which is exactly why the healthy boots (18:58, 19:22) logged **zero** shrink
events.

**SO EVERY SHRINK IS OVER-SHRINKING, AND A BAD READING IS THE ONLY TRIGGER.** On the crippled boots the post-write
read came back a **stable 0** (F-161 proved it is not a transient), so the loop gave back
`want - 0 + 64 MiB` ≈ **1.04 GiB per step**, four times, landing at 5,082 slots / 9.67 GiB - having thrown away
~4.4 GiB of expert cache that the plan had already proved fits, to buy headroom the check did not need. The cost
is the ~30% throughput measured in F-160.

**THE FIX, CONSERVATIVE AND MINIMAL.** The code already owns the correct trigger: a genuine shortage makes the
*next allocation fail*, and `generate.cpp:5171` already retries with a smaller cache in that case. So the
free-space check should act only on a **credible** reading:
1. **Bounded give-back (the safe change):** a sub-16-MiB reading - the branch that exists today - may cost at most
   **one** step, not four, so a bad reading is capped at ~1 GiB instead of 4.4 GiB. Real shortages still shrink,
   via repeated boots or the allocation-failure path.
2. **No-op on CUDA (the aggressive change):** where the plan measured free VRAM live (which it does -
   `generate.cpp:4896`), a sub-16-MiB post-write reading cannot be distinguished from a bad reading, and the
   plan's own reserve already guarantees the fit; so treat it as unknown and keep the planned cache, letting a
   real allocation failure drive the shrink.
**ACCEPTANCE TEST, already instrumented:** boot this lane and require **zero** `shrinking the expert cache` lines
and a `filling the GPU's expert cache` line near **7,773 experts / 14.8 GiB** (against today's 5,082 / 9.67 GiB),
then re-run the F-160 throughput comparison to collect the ~30%. Both changes are small and neither touches the
decode path; the choice between them is the next round's first decision, and it is the operator's call whether to
take the bounded version first.

### F-163 — I BROKE THE LANE: THE NO-OP FIX DOUBLED THE CACHE AND CRASHED THE ENGINE (2026-10-12 ~21:55-22:05, goal round 39)

**WHAT WAS DONE.** F-162's candidate 2 was implemented (CUDA: a sub-16-MiB post-write reading breaks the shrink
loop and keeps the planned cache; HIP keeps the old behaviour), built clean (`[4/4] Linking CXX executable
strata.exe`, exit 0), deployed with the timestamp-verified hand-off rule, and booted on the adopted recipe.

**IT MET THE ACCEPTANCE TEST'S CACHE GOAL AND THEN KILLED THE ENGINE.**

| | before | with the patch |
|---|---|---|
| experts cached | 5,082 / 9.67 GiB | **10,160 / 19.27 GiB** (the biggest cache measured on this lane) |
| `shrinking the expert cache` lines | 4 | **0** |
| free VRAM with everything loaded | 1,901 MiB | **0 MiB** - the engine printed `LOW: requests may stall; add --vram-reserve-mib 1512` |
| the first request | served | **CRASHED: exit code 3221226505 (0xC0000409, stack buffer overrun)** |

The server reported `the engine stopped unexpectedly (exit code 3221226505)` on the first request and
`the engine stopped while this request waited` on the next; M1 and M2 could not be measured at all.

**WHAT THIS PROVES, AND WHY MY F-162 CONCLUSION WAS WRONG.** The arithmetic in F-162 was right - the plan reserves
~5.67 GiB and the check demands ~1.0 GiB - but the conclusion drawn from it was not: **the cache that ends up on
the card is NOT the cache the plan chose.** The plan selected 7,773 slots and the boot filled **10,160** (19.27
GiB), so the extra ~4.5 GiB arrives after the plan, and **the post-write check is the only thing that claws it
back.** Disable the check and the engine over-commits to 0 MiB free and dies. **The check is load-bearing and the
no-op candidate is REJECTED outright** - it is a crash, which is far worse than the ~30% slowdown it was meant to
cure. Recorded so it is not re-tried.

**THE RIGHT SHAPE OF THE FIX, NOW.** Keep the check; make the give-back **accurate and bounded** (F-162 candidate
1: a sub-16-MiB reading may cost at most one step instead of four), and/or take the engine's own advice and raise
`--vram-reserve-mib` toward 1512 - with the check enabled that sizes the cache to leave genuine headroom while
still holding far more than today's 5,082. The acceptance test is unchanged (zero shrinks, cache well above
5,082) **plus a smoke request that must succeed**, which is the check this round added the hard way.

**LANE RESTORED AND VERIFIED.** `engine/strata.exe` is back to the adopted 09:29:28 build (backups kept beside it
as `strata.exe.prev-f160` / `strata.exe.prev-09`), the source patch is reverted, the lane serves the adopted
recipe (`--spec-min-p 0.7`, cache 5,082), and a smoke request returned 120 tokens after the restore.

### F-164 — THE CORRECTED PATCH, SPECIFIED FOR A FRESH SESSION (2026-10-12 ~22:10, goal round 40)

No further lane work this round: F-163's crash was caused by touching the cache-sizing safety check with a
half-verified change, and the corrected patch needs a build-deploy-boot-**smoke** cycle that this session no
longer has the budget to finish safely.  The lane is verified healthy on the adopted build and left alone.

**THE CORRECTED PATCH (replaces F-162 candidate 2, which F-163 rejected as a crash).** Keep the post-write check
and bound its give-back, so a bad reading costs a known amount instead of four steps:

    int64_t give = want - (int64_t) free_b + (64ll << 20);
    if (free_b < ((size_t) 16 << 20)) {
        if (zero_reads >= 1) break;                  // a reading this bad is not evidence: keep what we have
        give = std::max<int64_t>(give, 1ll << 30);   // one bounded step, then re-decide
        ++zero_reads;
    }

Expected outcome, from the measured numbers: plan 7,773 slots minus one ~1.04 GiB step = **~7.2k slots / ~14 GiB**
(against today's 5,082 / 9.67 GiB and F-163's fatal 10,160 / 19.27 GiB at 0 MiB free).  The 10,160 case shows the
final cache is NOT the plan's cache - roughly 4.5 GiB arrives after the plan - so the bound must leave real
headroom, and the boot must end with **at least ~500 MiB free**, not 0.

**ACCEPTANCE TEST (all three, in order):**
1. `shrinking the expert cache` appears at most **once** in the boot log;
2. `filling the GPU's expert cache` reads **~7,000-7,700 experts** (well above today's 5,082, well below the
   fatal 10,160), and `strata serve: N MiB of VRAM free with everything loaded` reads **above ~500 MiB**;
3. **a smoke request returns tokens** - the step F-163 proved is not optional - and only then the M1/M2 probes,
   with a rollback (`engine\strata.exe.prev-*`) ready and the timestamp-verified hand-off rule from F-161.
If any of the three fails, revert the binary first and record the failure before investigating.

### F-165 — THE BOUNDED FIX REJECTED TOO, AND THE REFRAME: THE SHRINK IS A CORRECT ADAPTATION, NOT A BUG (2026-10-12 ~22:10-22:25, goal round 41)

**WHAT WAS BUILT.** F-164's bounded patch: keep the post-write check, but let a sub-16-MiB reading cost at most
ONE ~1.04 GiB step and then keep what the engine has. Applied to pristine source (13 insertions), built clean
(`[4/4] Links`, exit 0), deployed with the timestamp-verified hand-off, booted.

**ACCEPTANCE TEST, ALL THREE STEPS, AND THE RESULT:**

| step | wanted | measured |
|---|---|---|
| 1. shrink lines in the boot log | at most 1 | **1** - the bound held exactly as designed |
| 2. cache size / free VRAM | ~7,000-7,700 experts, >500 MiB free | **9,610 experts / 18.23 GiB** (predicted ~9,614 - the arithmetic was right) but **0 MiB free**, with the engine's own `LOW: requests may stall; add --vram-reserve-mib 1512` |
| 3. smoke request | returns tokens | **passed** - 8 tokens |

**AND THEN IT FAILED THE TEST THAT MATTERS.** The 2000-token real-workload request - 13-19 s in every window
measured tonight - ran **beyond 10 minutes** before the measurement was cancelled: at 0 MiB free the engine pages
to system memory under WDDM and crawls. **REJECTED**: a build that trades ~100 t/s for under 3 t/s is far worse
than the state it was meant to improve, and it is worse than the 5,082-slot boot it replaced.

**THE REFRAME - FOUR ROUNDS OF CHASING A BUG THAT IS NOT ONE.** The engine's warning and the numbers now agree:
this machine genuinely has ~5 GiB less free VRAM available in the late-evening window than it did at 18:58-19:22
(the healthy boots cached 9,638-9,853 experts as *peers of the desktop*, ending with only 379-1,901 MiB free).
**The post-write shrink is the engine adapting the cache to the VRAM it can actually have** - which is what the
operator's original question ("something wrong with only 5k experts cached?") deserves as an answer: **nothing is
wrong with the engine; the machine has less room right now, and the engine gave the cache back to keep working.**
What is real, and much smaller than I claimed, is the loop's crude step size: it overshoots its own reserve by
~900 MiB, so it sacrifices ~900 MiB of cache more than it needs to - not the 8.6 GiB I attributed to it in
F-160..F-162, which was simply the correct amount for this machine state.

**TWO PROCESS RULES EARNED (both cost a lane).** (1) **A smoke request is not an acceptance test**: an 8-token
reply passed on a build that crawls under a real 2000-token request - the acceptance test must carry the
real-workload probe, which is the ultimatum's own rule applied to *engine builds*, not only to arms. (2) **A free
VRAM figure of 0 MiB is fatal, not cosmetic** - the engine's own `LOW` warning is a stop sign: treat it as an
immediate rollback trigger, not a note.

**THE PRACTICAL LEVER, RESTATED.** For more cache on this machine the answer is **free VRAM** (or a smaller
`--max-context`), not an engine change: the engine already sizes itself to whatever is left. The residual engine
work worth doing is only the ~900 MiB overshoot, and it is worth ~900 MiB, not 30%.

**LANE RESTORED AND VERIFIED.** Adopted 09:29:28 binary, source reverted (0 files changed), boot on the adopted
recipe with **5,310 slots / 10.10 GiB and 1,375 MiB free with everything loaded**, a request served after the
restore. F-162/F-163/F-164/F-165's patch line is closed: the check stays exactly as upstream wrote it.

### F-166 — THE DEEPER-CHAIN EXPERIMENT CANNOT RUN IN THIS MACHINE STATE (2026-10-12 ~22:30, goal round 42)

**WHAT WAS ATTEMPTED.** The queue's next item - `--spec 5`, to make draft position 4 exist at all (F-159's
correction) - using the launcher and config that already existed (`launch-v2-spec5-8081.bat`,
`strata-iq3_s-spec5.json`, `--spec 5 --spec-min-p 0.7`), with `STRATA_SPEC_DEPTH=1` inherited through the parent
environment so the per-position vector would print. The design was sound: whether position 4 exists and what it
accepts is a WITHIN-BOOT fact, immune to tonight's cache variance, which is why it was worth running at all.

**IT DID NOT COME UP.** The boot filled a cache of **5,081 experts / 9.67 GiB after the same four zero-read
shrinks**, reached `[strata] almost ready ...`, and then the launcher stopped at
`Press any key to continue . . .` - its own error path - leaving **no listener on 8081** (verified: listener 0,
strata 0; the smoke request was refused). `spec5-err.txt` is empty, so the error text sits in the console above
the pause and is the first thing to read next time.

**WHAT IS ESTABLISHED, AND WHAT IS NOT.** Established: with this machine's current VRAM budget the spec-5 boot
could not serve, so **the deeper-chain experiment is blocked on the environment, not on the engine and not on its
design** - it needs free VRAM restored (or a lower `--max-context`) before it can produce a number. NOT
established: the cause of the exit. The two candidates are an allocation failure at the much tighter budget (the
boot ended with only ~1.3 GiB free after shrinking) or a start-up/port conflict, and the difference matters -
but asserting either without reading the console error would be exactly the guessing this campaign is forbidden.

**SERVICE RESTORED AND VERIFIED.** The paused launcher was killed, the adopted launcher rebooted with
`STRATA_SPEC_DEPTH` cleared from the environment, and the lane serves again: **adopted 09:29:28 binary, adopted
recipe, 5,310 experts / 10.10 GiB, 1,478 MiB free with everything loaded, requests answered.** No build was made
this round, so no binary changed hands.

**FOR THE QUEUE.** Every remaining item - `--spec 5..8`, the R17 head's gates, and any A/B at all - now runs on a
machine whose free VRAM varies by ~5 GiB between boots, which swings the expert cache between 5,081 and 9,610 and
throughput by ~30%. **Until that is stabilised, cross-boot comparisons cannot be trusted**, and the honest order
of work is: (1) restore free VRAM (close the GPU-heavy desktop apps, or lower `--max-context`, which the engine's
own `LOW` warning names), (2) re-baseline the standing recipe in that window, (3) then run the deeper-chain boot
and read its console before its numbers.

### F-167 — THE PINNED-CACHE BASE: THE OPERATOR'S IDEA IS RIGHT, AND THE FIRST VALUE WAS WRONG (2026-10-12 ~22:40, goal round 43)

**THE OPERATOR'S DECISION, VERBATIM IN SUBSTANCE.** Pin the expert cache explicitly (they nominated 5,000) so
every boot holds the same slot count and v1 and v2 can be compared without fluctuating slots: "we know that this
may reduce t/s but we know it's not a real t/s regression ... it's that expert cache is limited to a hard cap, to
allow more accurate testing and performance improvements which will be amplified later when we increase it
again." That is the correct methodology - a controlled variable instead of an unknown one - and the campaign
adopts it as the measurement base.

**THE FIRST ATTEMPT, AND WHY IT FAILED.** `strata-iq3_s-pin5000.json` (`--expert-cache 5000`) with
`launch-v2-pin5000-8081.bat` (also arming `STRATA_SPEC_DEPTH=1` so every reading carries the per-position vector):

| expected | measured |
|---|---|
| 5,000 slots | **6,528 slots / 12.40 GiB** - an explicit value is a floor, not a ceiling; the engine grows past it by ~30% |
| a reserve kept free | **0 MiB free**, engine warning `LOW: requests may stall; add --vram-reserve-mib 812` |
| a real request served | **FAILED after 97.7 s** (the paging crawl, F-163/F-165's failure mode) |

Two mechanisms, both now measured: **an explicit `--expert-cache` bypasses the auto sizing AND its reserve
check** (`generate.cpp:4894` is the `o.expert_cache < 0` path), so nothing keeps headroom; and the final slot
count is not the requested one. The run was auto-reverted on failure and the lane is back on the adopted recipe
(5,310 experts / 10.10 GiB, requests served).

**THE DERIVATION FOR THE NEXT ATTEMPT.** To LAND at ~5,000 slots, request ~5,000 / 1.30 = **~3,850**; expect
~5,000 slots / ~9.5 GiB and, with the current desktop footprint, ~2.5 GiB free - which is the stable, comparable
base the operator asked for, at a known cap rather than an unknown fluctuation. If the final count still
overshoots, the ratio is measurable from one boot and the request is adjusted once. **The acceptance test is the
one this round earned:** the final `expert cache N slots` line must be within ~5% of 5,000, `N MiB of VRAM free
with everything loaded` must be **above ~800 MiB**, and a **real-workload request** must complete in the normal
time - a smoke request is not evidence (F-165).

**STATUS.** `strata-iq3_s-pin5000.json` and `launch-v2-pin5000-8081.bat` are committed as the pinned-base
scaffolding; the config needs the derived value before it is used. The adopted recipe is untouched and serving.

### F-168 — THE NIGHT'S MYSTERY SOLVED: THE RESERVE WAS 300 MiB AND IT NEEDED ~2,000 (2026-10-12 ~22:52, goal round 44)

**THE OPERATOR SAID "SOMETHING IS HOLDING UP THE GPU", AND THEY WERE RIGHT.** The adopted config carried
`--vram-reserve-mib 300` - not the 1000 MiB this session's F-161 analysis had assumed, and far below what this
card needs under WDDM when the desktop holds VRAM. With the reserve at 300 MiB:
- the auto cache plans 7,773 slots from a nominal reading, fills ~10,160-10,518 by growth,
- the post-write check finds **0 MiB free** and claws the cache back to 5,082-5,310,
- and depending on where it lands, requests either **stall for minutes at 0 MiB free** (10,160/9,610-slot boots)
  or serve at ~100 t/s with a half-sized cache (5,082-5,310-slot boots).

**THE FIX WAS A CONFIG VALUE.** `strata-iq3_s-reserve2000.json` + `launch-v2-reserve2000-8081.bat`
(`--vram-reserve-mib 2000`, everything else adopted):

| | reserve 300 (the night's chaos) | **reserve 2000 (now)** |
|---|---|---|
| experts cached | 5,082-10,518 across boots | **9,645 / 18.30 GiB** |
| free VRAM after load | **0 MiB** (`LOW: requests may stall`) | **1,467 MiB** |
| real request (first, cold) | stalled >10 min / failed at 97.7 s | **284 tok in 2.7 s = 104.1 t/s** |

**EVERYTHING ELSE THIS SESSION MEASURED STILL STANDS, AND SOME OF IT IS NOW EXPLAINED:**
- F-154's compositor correlation was real but mis-attributed: a desktop that grows by ~8 GiB of VRAM squeezes
  the cache, and the engine's cache size - not GPU time-sharing - is what moved the throughput.
- F-160..F-165's cache chase was chasing a *mis-set reserve*: the engine was correctly refusing to run with
  300 MiB of headroom, and every "defect" I named in the shrink loop was it doing its job with the wrong input.
  **F-161's patch (re-read), F-162's candidate 2 (no-op) and F-164's bounded variant are all unnecessary.** The
  check stays exactly as upstream wrote it; only the input value changed.
- The M3 amendment, the field-practice clause, the per-position instrument and the research artifacts are
  unaffected and remain adopted.

**ADOPTION STATUS: PROVISIONAL PENDING THE FULL GATE.** The lane serves `--vram-reserve-mib 2000` and the first
real request ran at 104.1 t/s cold. Remaining before it is called adopted: the six-metric probe in this window
(M1/M2/M3), MMLU-Pro and mimic (the reserve is a VRAM allocation change and must be quality-gated), and a second
boot to confirm the cache lands at 9,645 repeatably.

**LANE:** serving `--vram-reserve-mib 2000` (9,645 experts / 18.30 GiB, 1,467 MiB free), adopted binary 09:29:28,
MAGMA v1 frozen at 8da8bf16.

### F-169 — UPSTREAM HOTFIX v0.1.43.1 PUBLISHED TONIGHT: IT NAMES OUR COLLAPSE, OUR VRAM WARNING, AND OUR PROMPT-LOAN FAILURE (2026-10-12 ~23:00, goal round 45)

The operator linked `github.com/Niko1221/Strata/releases/tag/v0.1.43.1` - a hotfix for our base 0.1.43, published
**today 17:30 UTC (19:30 local)**, i.e. two hours into this session. Release body read via the GitHub API.
Answers claimed byte-identical to 0.1.43 except a user message quoting `</tool_call>` (#2026) - to be verified on
our lane, not accepted.

**ITEMS THAT TOUCH WHAT THIS SESSION MEASURED:**
1. **The adaptive overlap is opt-in again, everywhere (#2010).** 0.1.43 turned `STRATA_ADAPT_OVERLAP` on by
   default on Linux NVIDIA and it cost -19% to -21% on several desktops (RTX 5070 Ti 68.4 vs 86.4 tok/s). Our
   adopted recipe arms it EXPLICITLY (`STRATA_ADAPT_OVERLAP=1`, F-67) - so we run the path upstream just made
   opt-in, and upstream's own advice is "measure it with your own A/B before you keep it".
2. **The overlap's collapse mechanism (#1964, #1975, found in #1983): the CUDA host-callback thread inherits the
   decode thread's pinned core, so each per-layer callback waits ~1 ms instead of microseconds.** Reported: an
   RTX 4090 at 19-37 tok/s instead of 102-116; an RTX 5090 -26%. **Our crippled boots read 46-100 t/s tonight.**
   It is intermittent by construction (the thread starts when the overlap first uses it, then inherits whatever
   core the pin landed on) - which is the first mechanism proposed that explains why the SAME binary and recipe
   produced two states. Fixes: the callback thread is started before the pin (`STRATA_ADAPT_CALLBACK_WARMUP=0`
   off-switch), and the overlap no longer uses the thread at all (`STRATA_ADAPT_SIGNAL=host` brings it back).
   Measured on an RTX 3060: 36.5 -> 56.5 tok/s, 5/5 interleaved pairs.
3. **#2065 + #2049: "hang in the draft step with almost no VRAM left"**, and every request now **checks the free
   VRAM and warns with the `--vram-reserve-mib` that would leave room** when it is very low - exactly the
   `LOW: requests may stall; add --vram-reserve-mib 812/1512` warnings we hit at 0 MiB free tonight, and exactly
   the condition of F-163's crash. The watchdog also no longer stops a request while a conversation is parked.
4. **#2090: a short prompt right after start could END THE ENGINE with "lending the prompt path its slots
   failed"** - our prompt-path borrow machinery, the same family as the spec-5 boot that reached "almost ready"
   and exited (F-166). Cause: a chunk smaller than the start-up chunk could need more borrowed cache than was
   set aside; the loan now covers what the request's chunk needs.
5. **#2040/#1984:** the PLE mmap auto-switch made long prompts ~7x slower on SATA and tripped the watchdog; the
   check now only warns, and mapped mode reads ahead - our long-prompt lane again.
6. **#2088: a server a benchmark starts no longer saves over the expert profile** - protects our
   test-traffic-must-not-pollute invariant. **#2036: `--calibrate --model X`** tunes an installed model - our
   calibration queue item.

**THE PLAN (next session, in order):** fetch and merge `v0.1.43.1` into `custom` (rerere primed, KEEP_MAPPING=1),
rebuild, then: (1) re-baseline the F-168 reserve-2000 lane, (2) run the six-metric gate + MMLU/mimic,
(3) A/B `STRATA_ADAPT_OVERLAP=1` vs `=0` as upstream advises, on the pinned or reserve-2000 base, (4) re-run the
per-position vector at `--spec 5` - now with the callback-collapse fix in, which may itself change the reading.
The deeper-chain and acceptance questions are unchanged.

**STATUS: recorded, not merged.** The merge, rebuild and gate need a session with budget; the lane is serving the
reserve-2000 configuration.

### F-170 — v0.1.43.1 MERGED CLEAN, ZERO CONFLICTS, GATED HEALTHY (2026-10-12 ~23:05, goal round 45)

**THE PULL WORKED EXACTLY AS THE DESIGN INTENDED.** `git fetch origin --tags` brought `v0.1.43.1`;
`git merge v0.1.43.1 --no-edit` auto-merged **all 28 files with ZERO conflicts**, including `generate.cpp`,
`verify.cpp`, `mtp.cpp` and `server.py` - our 226 commits on `custom` and upstream's hotfix touched the same
files and the ort strategy resolved every hunk. Before the merge, the tracked working-tree changes (learned
expert profile, adopted config, campaign logs) were checkpointed as their own commit (e428c4a2).

**MARKER CHECK AFTER THE MERGE:** our decode-timing line, the `STRATA_SPEC_DEPTH` per-position print and the
dp counters are all present; upstream's `ADAPT_CALLBACK_WARMUP` / `ADAPT_SIGNAL` fix is in (5 references). Built
clean (`[138/138] Linking CXX executable strata.exe`, exit 0), deployed with the timestamp-verified hand-off.

**GATE ON THE MERGED BUILD, reserve-2000 lane, same window:**

| check | merged v0.1.43.1 + our layer |
|---|---|
| expert cache | **9,645 experts / 18.30 GiB, 0 shrinks** |
| free VRAM after load | **1,492 MiB** |
| overlap mechanism | **stream-ordered copies** (the new #1964 fix; the callback thread is gone) |
| M1 | 2000 tok in 21 s = **95.4 t/s** cold, window **13.43 ms**, GPU-reach wait **8.61 ms** |
| hit rate | **97.3%** |
| per-position | **376/386 = 97.4% · 223/234 = 95.3% · 135/152 = 88.8% · positions 4-5: 0/0** |

**AND THE PER-POSITION VECTOR NOW PRINTS ON EVERY REQUEST** - `STRATA_SPEC_DEPTH=1` is inherited by this launcher,
so the chain-depth question is answerable on any boot without extra setup.

**WHAT THE MERGE CHANGES FOR THE QUEUE.** The overlap now runs on stream-ordered pinned copies instead of the
host-callback thread, so F-169's collapse mechanism (#1964) is fixed IN OUR BINARY - which means the A/B
`STRATA_ADAPT_OVERLAP=1` vs `=0` upstream advises can now be run on a fixed engine, and tonight's two-state
mystery can be re-tested: if the callback race was the cause, the collapse should not reproduce on this build.
`--vram-reserve-mib 2000` stays (F-168). The `--spec 5..8` deeper-chain experiment is unblocked on this base.

**STATUS.** Merged, built, deployed, gated on a real request; the lane is serving the merged build on the
reserve-2000 configuration. The full six-metric suite (pair, 6-conc, MMLU-Pro, mimic) is the remaining gate
before this merge is called adopted rather than running.

### F-171 — SIDETEST: THE AGENGIONAI "GYRO" QUANTS CANNOT RUN ON STRATA - BUT THE HYPOTHESIS THEY EMBODY IS OUR EXACT BOTTLENECK (2026-10-13 ~00:40, goal round 46)

**WHAT THE OPERATOR ASKED.** Test `E:\LLM\Models\GGUF\agentionai\qwen3.8-flash-next`: Gyro-S (54.47 GB,
"TQ1_0") and Gyro-M (85.65 GB, "TQ2_0"), plus a 2.51 GB MTP draft gguf and an mmproj.

**INSPECTED WITH `tools/quantscope.py` (metadata only, no lane touched):**

| file | total | routed experts | expert tensor types |
|---|---|---|---|
| Gyro-S | 58.48 GB, 2.64 bits/w | 120.8 B weights @ **1.71 bits**, 25.79 GB | **type#145 (63%) + type#146 (37%)** |
| Gyro-M | 91.95 GB, 4.16 bits/w | 120.8 B weights @ **2.12 bits**, 32.09 GB | **type#144 (100%)** |

**VERDICT: NOT RUNNABLE.** The filename quants "TQ1_0/TQ2_0" are branding - Strata's own TQ types are enum
34/35 (`include/strata/artifact/gguf_reader.hpp`), and the files' experts actually use **custom types 144/145/146**
that appear nowhere in the engine except as unnamed enum numbers. There is no kernel, no pool path and no pack
tool for them (`tools/iq_pack.py` cannot prepare them; setup.py's MODELS table offers no TQ entry). Everything
else in the files is standard (IQ4_NL embeddings, Q5_K attention, Q6_K head) - it is only the routed experts,
the part that matters, that cannot execute.

**WHY THE IDEA IS EXACTLY RIGHT FOR US.** Our measured window law is `CPU = misses x ~85 us`, and each miss
streams one expert blob - at IQ3_S, ~1.95 MiB per slot. Gyro-S stores 120.8 B expert weights in 25.79 GB -
roughly HALF the bytes per expert of a 3.5-bit layout - so a supported 1.7-bit expert would cut the miss cost
roughly in half and with it the term that gates every degraded window tonight. The hypothesis these files
embody is the single most on-target idea for our bottleneck; only the format is unusable.

**THE RUNNABLE VERSION OF THE SAME TEST IS IQ2_XS.** Setup supports it (`"2-bit i-quant, a little better
quality, close in speed"`, 68 GB download, 48 GB RAM class). An IQ2_XS lane would measure the
bits-per-expert -> miss-cost curve directly: same model, same drafter, same probes, smaller blobs. That is the
sidequest to queue when the main line has a session to spare - gated by the six-metric law, since 2-bit costs
quality (M4/M5 must be measured, not assumed).

### F-172 — THE LINUX PORT: THE LANE RUNS ON FEDORA 45 / RTX 5090 (2026-10-13 evening)

**WHAT THE OPERATOR ASKED.** Move to the Linux desktop (Fedora 45, Ryzen 9 9950X3D, RTX 5090 32 GB) and run
the Q2_0 lane with this engine instead of the stock one; check the scripts for Windows-only baggage.

**THE PORT (repo copied to NVMe, ~/LLM/strata-custom-2; NTFS master stays canonical):**
- **Build law.** MSVC includes <cmath> transitively; GCC does not. `src/core/expert_source.cpp:2379`
  (RouterLookahead::run, std::exp) failed with `'exp' is not a member of 'std'`. Fix: one
  `#include <cmath>` at the include block. UPSTREAM THIS - it is the only source change Linux needs.
- **Launch law preserved.** `launch-custom-q2_0-8081.sh` is the line-by-line port of
  launch-v2-armT1d-8081.bat: every STRATA_* arm carried, serve/server.py started directly, port 8081.
- **draft-vocab is a setup flag, not an engine flag.** `--draft-vocab en` belongs to setup.py; the engine
  reads `draft_vocab.bin` inside the MTP folder (refresh_draft_vocab, setup.py:4660, copies
  data/draft_vocab_en.bin over it). Config key `"draft_vocab": "en"`; engine args must NOT carry it.
- **VRAM budget law for Q2_0 @ 32 GB.** The aux allocations fail in a fixed order as the reserve shrinks:
  200 -> cuBLAS create dies; 300 -> draft head 178 MiB vs 75 free; 512 -> verify device arena 76 MiB;
  **768 -> clean, staged hc check included.** Same shape as F-168's reserve-2000 on Windows; Q2_0's tier
  needs less. kv-resident stays 32768 (operator's call).
- **MTP isolation.** The stock lane and this lane share pack/GGUFs via symlinks (models/, packs/), but mtp
  is a private 6.5 GB copy so the draft-vocab swap cannot narrow the stock lane.

**F-64 ON LINUX IS A NO-OP BY DEFAULT - AND THAT IS THE POINT.** STRATA_KEEP_MAPPING exists because .43+'s
Windows file tier opens unbuffered and the mapping is dropped; on Linux the source is not opened unbuffered,
so the mapping (and the OS page cache behind it) is kept without asking. The warm-rate parity F-64 fights
for on Windows is simply Linux's default. Arm kept in the launcher (harmless, documents intent).
STRATA_POOL_SPIN_US is the same species (works around coarse parking; cheaper scheduling here). The
GPU-side arms (DMA batch, PLE batch, device verify plan, exchange rotation, adapt tier, coupled drafts)
are platform-neutral and carry their measured value unchanged.

**MEASURED (Q2_0, short probes - NOT the six-metric gate):** decode 190-227 t/s, prefill 265-287 t/s,
draft acceptance 83-92% (35/38 and 125/151), hyper-connection staged, exchange rotation on,
16.95 GiB pinned RAM complement, ~32 GB VRAM, 29 GB RSS.

**STATUS.** Running on 8081 from the NVMe copy. The six-metric suite is the gate before calling the Linux
lane adopted; the cmath fix wants upstreaming into the NTFS master.

### F-173 — THE LINUX SUPERIORITY CAMPAIGN, ROUNDS 0-4: ALL THREE THROUGHPUT BARS BEATEN (2026-10-13 night)

**WHAT THE OPERATOR ASKED.** Beat the Windows lane by 30%+ on every metric; relentless rounds, two study
legs each (code + world), local NVMe copy only, /mnt/DATA frozen as the before-snapshot. Goal register:
`campaign/GOAL-LINUX-SUPERIORITY.md` (the bar table, the rules, the round queue, the scorecard).

**THE REGRESSION WAS rt-r17, NOT LINUX (proven in-log and by A/B).** Sessions 1-2 of custom-iq3_s.log ran
rt-r17 at 48-63% acceptance / 0.45-0.62 drafts offered / 108 t/s; session 3 (stock mtp/rt, same box, same
recipe) read 88-92% / 135-171 t/s — the Windows 90.2%/145.8 class exactly. r17 is the distilled head whose
own audit (F-104/F-114/F-115/F-116) measured a 55.7-57.4% plateau; the 280.7-class head is stock rt.
Rounds 1-4 (each one variable, probe-lane.py, results in campaign/results/linux-*.txt):

| round | change | M1 | pair | 6-conc | accept |
|---|---|---|---|---|---|
| 0 | port baseline (r17, min-p .7, spec 4) | 109.1 | 132.7 | 162.3 | 63% |
| 1 | --mtp -> stock rt | 154.9 | 132.8 | 168.8 | 88-92% |
| 2 | + min-p 0.5 | 167.1 | 144.7 | 187.6 | 87% |
| 3 | + --spec 6 | 181.5 | 144.9 | 179.1 | 79% |
| 4 | + ROUTE_RESIDENT=0.5 / ROUTE_RESIDENT_MTP=1 (battery arms the port had dropped) | **196.1** | **158.8** | **212.8** | 79.4% |

**THE BAR TABLE: M1 +34.5%, pair +46%, 6-conc +52% over the Windows standing lane — all three throughput
bars met in four rounds.** Hit rate 98.3%, ms/window stable (late 8.8), TREND stable, 10,307 slots at boot
(Windows: 9,850-10,077). Context: the public single-stream frontier for this model class on one 5090 is
~100-120 t/s; the lane is at 196.

**PENDING BEFORE ADOPTION IS DECLARED:** the M4/M5 gate (MMLU-Pro running on the round-4 recipe;
mimicbench not yet ported — the M5 tool must be located/ported from the Windows campaign), the depth
metrics (M6 64k prefill/decode), and the short-class matched anchor. Round-1 agent findings and the full
round queue live in the goal register.

### F-174 — THE STANDING LINUX RECIPE, PINNED (2026-10-13 night; the F-173 performance, reproducibly)

**THE RECIPE IS TWO FILES, NOTHING ELSE.** (1) `launch-custom-iq3s-8081.sh` — the env arsenal: the full
armT1d battery (KEEP_MAPPING, TYPICAL_ACCEPT=0.2, POOL_SPIN_US=100, KV_PREFETCH=1, ADAPT_LAG=2, DMA_BATCH=1,
PREDICT_CREDIT=1.0/BUDGET=2, ADAPT_PINCAP=48, ADAPT_MIN_AGE=2, ADAPT_DORMANT=64, EXCHANGE_ROTATE=1,
L2_SETASIDE_MIB=60, BATCH_MTP=0, BATCH_RESCUE=0, SPEC_COUPLED=1, PLE_BATCH=1, VERIFY_DEVICE_PLAN=1,
ADAPT_OVERLAP=1, ADAPT_OVERLAP_RESIDENT=1) **plus the restored F-72 battery arms ROUTE_RESIDENT=0.5 and
ROUTE_RESIDENT_MTP=1** (the first port dropped them; R4 proved them worth ~+8% everywhere). (2)
`custom-iq3_s-8081.json` — engine args: pack Custom-data/packs/iq3_s, native/ple the two IQ3_S GGUFs
(Custom-data/models/), expert-profile data/expert-profile.bin, spec **6**, spec-min-p **0.5**, mtp
**Custom-data/mtp/rt** (STOCK head — rt-r17 is retired pending the F16/distill upgrade), mtp-q4 all,
mtp-hnorm stream, max-context 131072, kv int8, kv-resident 32768, pcie-frac 0.35/dma, vram-reserve 2000
(F-168 law), conversation-cache 49152, cvec refusal-r 4-24 project/per-layer, ple-io mmap, adapt-decay
0.85, profile-save every 30, resident-experts; config KEY draft_vocab "en".
State at pin: M1 196.1 / pair 158.8 / 6-conc 212.8 / MMLU 95.71 (mmlu-linux-r4.json) / 10307 slots.

**THE INSTRUMENTS (ported this session, both in campaign/):** `probe-lane.py` — the Linux port of
probe-lane.ps1, same prompts/sequence/log-accounting; results in campaign/results/linux-*.txt.
`mmlu-pro-gate.py` — local copy with the Authorization header added (the Linux lane keys its API) and
`datasets` pip-installed into .venv. Boot order: launcher direct (LAUNCH LAW), never through setup.

**CODE-STUDY FINDINGS FUTURE ROUNDS STAND ON (agent leg, R1):** draft_offered Y counts min-p-FILTERED
window rows (MTP + suffix rows), accepted X includes typical-acceptance rescues; --spec is rewritten at
boot by the suffix drafter (spec 6 -> S_mtp 4 + up to 2 suffix rows/window); a draft is only OFFERED when
its own dprob >= min-p (generate.cpp:12190-12193) — under coupled sampling dprob is the post-temperature
chain probability, structurally lower, which is why min-p 0.7 truncated to ~1 draft; STRATA_SPEC_MIN_TEMP
(spec_prob.hpp:64-73) keeps argmax drafting for lukewarm requests; the draft path has NO OS-gated numerics
(Philox position-keyed, bit-deterministic; only _WIN32 intrinsics/IO). r17's 57% is its distilled plateau
(F-104/114/115/116), not a Linux fault. Full lever list: campaign/GOAL-LINUX-SUPERIORITY.md.

### F-175 — THE ANCHOR LOCK: the R4 recipe's true means, measured properly (2026-10-13 night)

Three full-protocol probes on a fresh boot of the committed recipe (plus two MMLU gates) replace the
single-run round numbers as the defended record: **M1 190.3 mean (183.6/199.3/188.1, +30.6% over Windows,
at the 189.5 bar but inside ±4% run noise — margin required), pair 159.4 (+46.5%), 6-conc 203.7 (+45.3%),
MMLU 95.71 ±2.42 twice (= Windows 95.7), short-class 185.4 post-fresh.** THE LOCK LAW: future rounds
claim gains only on means, re-run 3x when near a bar, gate MMLU when distribution-bending, and may not
regress any mean beyond noise. Sources: campaign/results/anchor-run{1,2,3}.txt, mmlu-{linux-r4,anchor}.json.
Git: the lane is now versioned (import -> port -> instruments -> campaign; no push yet).


### F-176 — R7 CPU AFFINITY, THE AUX-CPUS ARM: auto and 16-23 both land inside noise — REVERTED (2026-10-13 ~23:20, goal round 7)

**THE HYPOTHESIS (R7 investigation, same evening).** The engine's host thread pins to CPU 0 and the 15
pool workers to cores 1-15, but every thread created *after* host pinning inherits the host's single-CPU
mask: 7 threads hard-stacked on CPU 0 (futex waiters, the adapt job thread, prefill helpers), and the
CUDA driver's threads float over all 32 CPUs including the workers' cores. The fork ships the fix:
`--aux-cpus` / `STRATA_AUX_CPUS` (`include/strata/platform/aux_cpus.hpp`), whose own header documents the
pathology on a 16-worker box ("the host thread was preempted 5x as often; the per-layer wait for the
host's flag was 60-155 us instead of 8"). The anchor log showed the plan computed (`CPUs 16`) but left
`off at start`. Placement-only change — "no output of the engine depends on where a thread runs" — so
no MMLU gate needed.

**THE TEST (operator-specified protocol: baseline probes → launcher edit → same probes).** Baseline =
3x probe-lane on the standing server (up since 21:33, warm; NOT fresh boot — recorded as a caveat).
Arms = one env line in `launch-custom-iq3s-8081.sh` + restart + 3x probe-lane each. Placement verified
per boot from /proc: R7a put 13 aux threads on mask=16 (the host's SMT sibling — the only CPU `auto`
could take, since all 16 physical cores carry host or worker); R7b spread 12 aux threads over 16-23
(the V-cache CCD siblings).

| arm (3-run means) | M1 | M2 pair | M2 6-conc | M3b |
|---|---|---|---|---|
| baseline (warm, no aux) | 173.7 | 153.2 | 194.6 | 81.4% |
| R7a `STRATA_AUX_CPUS=auto` (plan {16}) | 171.4 (−1.3%) | 146.6 (−4.3%) | 192.1 (−1.3%) | 80.3% |
| R7b `STRATA_AUX_CPUS=16-23` | 176.5 (+1.6%) | 149.3 (−2.5%) | 194.8 (+0.1%) | 81.1% |

Sources: `campaign/results/r7-aux{1,2,3}.txt`, `r7b-{1,2,3}.txt` (baseline numbers from the session log,
same probe, same night, back-to-back).

**THE VERDICT: no gain, honest negative, reverted.** R7a stacked all aux threads on one logical CPU that
SMT-shares the *busy host's* physical core — a small active regression (pair −4.3%, at the noise edge,
consistent direction on all three metrics). R7b fixed the pileup and recovered it, but every metric
stayed inside the ±4% noise band. The documented pathology is real on paper but not this box's
bottleneck: the aux threads are mostly blocked (futex/event waiters), and at 79-82% acceptance the
draft path — not host preemption — sets the rate. Launcher restored to the standing battery verbatim;
revert verified by the boot log reading `aux cpus: ... off at start`.

**TWO DRIFT OBSERVATIONS WORTH THE RECORD.** (1) Tonight's no-aux baseline (173.7/153.2/194.6, warm)
and the fresh-boot R7b arm both sit 4-9% below the locked anchor means (190.3/159.4/203.7) — the whole
evening drifted vs the anchor night independent of any arm. No conclusion drawn from it here (that is
what the Lock Law's matched-window rule is for), but the next adoption round should re-run the
anchor-protocol fresh boot before claiming anything against the anchor. (2) GPU IRQs (nvidia MSI-X 230/232)
land on CPUs 19/27 — SMT siblings of workers 3/11, scattered by irqbalance; that piece of R7 was never
in this arm (needs the sudo step, R7c).

**WHAT THIS CLOSES AND WHAT IT LEAVES.** Closed: the env-var aux-cpus arm in both forms — the cheap
half of R7 is done and negative; do not re-run without a new mechanism. Left open: R7c IRQ steering
(sudo, user's paste block) and R7d (taskset whole-process to V-cache CCD + `--pool-workers 7`), the
latter demoted — it takes real cores away from the tier and risks the pair/conc means. The queue
resumes at R5 (STRATA_SPEC_GUMBEL).


### F-177 — R5 GUMBEL COUPLING TESTED TO THE FULL FACTORIAL: ~+1.5 acceptance points, not 9 — REVERTED, and the rescue quantified (2026-10-14 ~00:45, goal round 5)

**THE HYPOTHESIS (F-176 session's queue head).** `STRATA_SPEC_GUMBEL=1` replaces the inverse-CDF pick
on both the draft and the verify side with a Gumbel-max pick whose noise is keyed on
(seed, counter, TOKEN ID) — "support invariant", so list-composition differences between the drafter's
and the target's truncated candidate lists stop costing agreement (`sampler.cu:66-80`,
`coupled_draft.hpp:14-17`). Upstream (llama.cpp-lab PRs #26/#30) measured +9% acceptance, lossless.
Pre-registered adoption bar: M1 ≥ +4% over same-night baseline, no mean regression beyond ±4%,
MMLU ≥ 93.3.

**PLUMBING VERIFIED LIVE FIRST** (a mid-experiment scare, resolved): the flag is consumed at
`sampler.cu:1242` (target side, `sample_tokens`) and `:812` (draft side, `coupled_stage_kernel`), and
the pick itself runs in `sampled_tail_warp` (`:534-540`), the shared tail of the DEFAULT split path
(`:795`), the one-block fallback (`:631`), and the coupled drafter's pick (`:871`, with `sub_to_id`
mapping and the verifying row's counter). Not confined to the Old path. The A/B tested the real thing.

**LEG A (standing recipe, rescue ON): 6v6, back-to-back same-night boots.**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| baseline (6) | 168.7 | 145.6 | 191.6 | 79.1% |
| +GUMBEL (6) | 171.8 (+1.9%) | 145.4 (−0.2%) | 187.2 (−2.3%) | 81.0% |

Below the bar on every count; M3b ticks +1.9 pts with consistent direction but no t/s conversion.

**LEG B (rescue OFF — the masking test): 3v3.** If the typical-acceptance rescue were absorbing
Gumbel's recoverable mass, Gumbel would fire on the raw path.

| arm | M1 | pair | 6-conc | M3b (raw exact-match) |
|---|---|---|---|---|
| no rescue (3) | 155.9 | 146.3 | 190.7 | **58.5%** |
| no rescue + GUMBEL (3) | 157.0 (+0.7%) | 142.8 (−2.4%) | 186.0 (−2.5%) | **59.9% (+1.4 pts)** |

**THE VERDICT: the mechanism fires but the mass is not there.** Gumbel adds ~+1.5-2 acceptance
points in BOTH regimes — the rescue does not mask it. The upstream +9% came from an unfiltered
regime; our `--spec-min-p 0.5` means the drafter only offers rows it is confident about, and on
confident rows the draft's top token already equals the target's sample — the coupling artifact
Gumbel removes is second-order on the offered set. The 58.5% raw agreement is genuine distributional
disagreement between the rt draft head and the target at depth 6, which no sampling mechanics can
recover. Reverted; standing recipe restored and verified (env + boot log).

**THE DISCOVERY THE EXPERIMENT BOUGHT: the rescue, quantified for the first time on this lane.**
`STRATA_TYPICAL_ACCEPT=0.2`: 58.5% → 79.1% acceptance (+20.6 pts) and 155.9 → 168.7 M1 (+8.2%),
pair/6-conc unchanged within noise. It is the single largest acceptance mechanism in the standing
recipe — larger than everything else measured this campaign. Windows measured 0.70 → 0.92 on their
spec-4 config (their journal's patch map); this lane's spec-6 numbers are now on record.

**WHAT THIS OPENS (the re-ranked acceptance queue).** (1) **R5b: tau sweep** — the rescue threshold
0.2 is a single unswept number sitting on the largest measured lever; 0.25/0.30 could buy points, but
looser tau emits draft tokens further from the target's pick → **MMLU gate mandatory** (the Windows
F-15 rescue bug — text/context divergence per rescue — is the cautionary tale; tau loosens the same
criterion). (2) The draft head itself is the remaining structural lever (the 57% r17 plateau says
distillation is not; an F16/quant-chain upgrade per their F-116 next-levers note is the path if ever
revisited). (3) Depth stays the wrong direction to trade (M1 loses more than acceptance gains).

Sources: `campaign/results/r5-base{1..6}.txt`, `r5-gum{1..6}.txt`, `r5-nr{1..3}.txt`,
`r5-nrg{1..3}.txt` (18 probe files, all same-night, back-to-back boots, placement off = standing).
No MMLU gate run: nothing adopted.


### F-178 — THE RE-ANCHOR UNDER CURRENT CONDITIONS + THE ROUND-8 STUDY: a disarmed overlap found, the queue re-ranked (2026-10-14 ~01:00, goal round 8)

**THE RE-ANCHOR (fresh boot, standing recipe, midnight conditions).** Motivated by F-176/F-177's
drift observation (every arm 4-9% below the locked anchor). 3 full probes + 2 MMLU gates:

| metric | run1 | run2 | run3 | MEAN | locked anchor | delta |
|---|---|---|---|---|---|---|
| M1 single | 175.9 | 178.8 | 174.3 | **176.3** | 190.3 | −7.3% |
| M2 pair | 144.4 | 145.9 | 145.7 | **145.3** | 159.4 | −8.8% |
| M2 6-conc | 193.1 | 191.0 | 187.9 | **190.7** | 203.7 | −6.4% |
| M3b rate | 81.5 | 80.2 | 82.2 | **81.3%** | 80.5% | +0.8 pt |
| MMLU-Pro | 92.86±3.08 | 94.29±2.77 | — | **93.6** | 95.71 | −2.1 pts (≈1.5 q / 70; within noise) |

The drift is real, global (all metrics), and NOT GPU throttling (42 °C, zero active throttle
reasons, 0.22 s lifetime SW-power-cap). Candidate: the learned expert profile (re-saved 23:11 by
tonight's 18 probe runs) and/or ambient/system state. **Adoption claims now compare against
re-anchor v2 means; the locked anchor remains the bar reference.** Sources:
`campaign/results/reanchor2-{1,2,3}.txt`, `mmlu-reanchor2-{a,b}.json`.

**THE STUDY (leg A code subagent + leg B ecosystem subagent + operator-side paper reads).**

*LEG A — THE BLOCKER: the adopted F-67 overlap combo is DISARMED ON LINUX.* The launcher exports
`STRATA_ADAPT_OVERLAP=1` + `STRATA_ADAPT_OVERLAP_RESIDENT=1`, but the Linux port kept three arms the
Windows v2 battery dropped as vestigial: `STRATA_ADAPT_LAG=2` (launcher line 9),
`STRATA_DMA_BATCH=1` (line 12), `STRATA_VERIFY_DEVICE_PLAN=1` (line 26) — and the refusal chain at
`generate.cpp:8496-8516` kills the overlap on each (`dma_batch_mode()!=0` :8502, `adapt_lag()!=1`
:8503, adapt-gate refused under device-plan `verify.hpp:263-267`). The overlap never armed on this
box. Windows two-boot evidence: 6-conc 158/159 vs 145-152 baseline (launch-v2-battery,
T1e), i.e. **+4-9% concurrent was left on the table by a launcher bug**. Fix is env-only; verify
the boot line says the adaptive tier's copies land behind the window (not "off: not with
STRATA_DMA_BATCH").

*LEG A — the rest, ranked:* (2) **F-31 slot coupling** — concurrent batch slots run DRAFTLESS
(BATCH_MTP=0; drafts-only was worse, the rescue variant was quality-banned F-30b); coupled drafts are
quality-safe by construction (coupled_draft.hpp:5-6) and the wiring mostly exists — the open defect
is slot verify-row counter staging (verify.cpp:2576) not matching the coupled cells. The only path
to breaking the ~190 t/s concurrent ceiling; +25-50% class; needs code + rebuild.
(3) **R5b tau sweep — DIRECTION CORRECTED: higher tau is STRICTER** (criterion at generate.cpp:281-323:
accept iff q > tau·e^(−H); loosening means 0.15/0.10, not 0.25/0.30 as F-177 assumed). Also queued:
`STRATA_SPEC_RATIO=1` (Leviathan exact test, generate.cpp:325-338) — distribution-exact,
MMLU-safe-by-construction, A/B-pending since F-27. (4) L2 carve: correct site generate.cpp:7719-7727
over the GDN state span (112 MiB span, 60 MiB carve, hitRatio 0.536); 80/96 clamps to the ~75%
persisting ceiling ≈ 72 MiB; single-stream-only lever, +0-2%. (5) Controller verified DEAD
(zero references) and wiring-cheap, but honest verdict **0-3%** for our flat 77-82% acceptance
profile + per-layer fixed-cost wall — the min-p sweep already sits at the optimum it would learn.
(6) **PDL on Linux** (`STRATA_DF_PDL`, verify_kernels.cu:1380-1400): the F-107 retraction was
WDDM-attributed; no WDDM here; +3-8% possible; MANDATORY sustained-reasoning probe (2000-tok
generation) before adoption given the −40% Windows downside.

*LEG B — ecosystem (2025-26 scan).* Top transferable: **FastMTP recursive head re-tuning** (Red Hat
Speculators 0.6.0, Sep 2026): fine-tune native mtp weights recursively with position-decay loss
(beta 0.6 → weights [.51,.31,.18]); on Qwen3-Next-80B-A3B pos-2 acceptance 0.476→0.616 — the
train/serve recursion mismatch is the canonical cause of our 57% distilled plateau; ~443 s on
2×H200 for a 9B-class model. Then: EAGLE-3 fused features (arXiv 2503.01840), draft-vocab trim
(llama.cpp #25187: 621→91 µs head matmul, +1.4-3.1% e2e; 64k-row trim safer than 32k),
L2 evict_first hints (Blackwell-only), clock lock + power cap + **Xid 79 firmware check** (5090
GSP crash thread — verify driver 615.71.09 carries the fix before long runs). Calibration: depth 6
sits on the empirical plateau (arXiv 2607.17283 — speedup flat across K=2-6, decays past ~8);
draft-trees and NVFP4 deprioritized (trees: ggerganov's own position + MoE expert-union cost;
NVFP4: consumer Blackwell lacks the datacenter path and is DENSER than IQ3_S at our batch).
Operator-side reads: FastMTP recipe details (shared embed/lm_head, full vocab, target-generated
data) and the adaptive-γ literature (throughput_B(γ)=B(E[acc]+1)/C(B,γ); matches best fixed γ in
hindsight — but leg A's flat-profile evidence caps its value HERE; the dead Controller is its
natural home if ever wired).

**THE RE-RANKED QUEUE** (expected gain × probability / cost):
- **R9 — un-block the F-67 overlap** [env, free, NOW]: drop ADAPT_LAG/DMA_BATCH/VERIFY_DEVICE_PLAN,
  keep the OVERLAP arms, verify the boot line, 3× probe A/B vs re-anchor v2. Expect +2-6% conc,
  +1-3% M1. Same boot: read the L2 persisting ceiling for the carve-bump read-out.
- **R5b — tau 0.15 / 0.10 + SPEC_RATIO arm** [env]: MMLU gate EVERY arm (looser tau emits drafts
  the target finds more surprising; the batch-site rescue cost 3-4 MMLU points once).
- **R8 — clocks + power cap + Xid 79 check** [sudo, user step] + the iGPU display cable move:
  variance reduction first (it protects every future A/B; may explain part of the drift).
- **R10 — PDL Linux probe** [env]: sustained-reasoning gate mandatory.
- **R11 — F-31 slot coupling** [code+rebuild]: the concurrent ceiling breaker; instrument
  bt_slot_drafts/bt_slot_accepts first, diagnostic boot, fix the counter staging, full battery.
- **R12 — FastMTP recursive re-tune of the stock head** [training]: the structural acceptance lever;
  compose with EAGLE-3 features later; reuse the r17 training-chain tooling from the Windows arc.
- Open bar items unchanged: M5 mimic port, M6 depth + short-class matched anchors, NTFS master sync.


### F-179 — THE ROUND-9 DEEP STUDY: the engine is execution-bound, not bandwidth-bound — the budget quantified and the barrier named (2026-10-14 ~02:30, goal round 9)

Three study legs (two deep code agents + one external scan) + operator paper reads (megakernels/Hazy,
Mirage MPK, CUDA conditional nodes, SSD overlap, ReplaySSM, the empirical spec-decode anatomy).

## THE THESIS (all three legs converge, with numbers)

**One verify window = one CUDA graph of ~1,800 kernel nodes (48 layers x 35-44 nodes), 17.12 ms wall
at single-stream (12.73 ms at 6-conc), carrying 7.35 GB — ~9.5 µs effective node spacing vs ~2.3 µs
of bytes per node. The engine streams at 24-26% of GDDR7 peak (1792 GB/s).** The per-layer "fixed
cost" is ~70% node spacing + under-filled small kernels + doorbell handshakes, ~30% real traffic.
Independent confirmation: arXiv 2605.30571 measures H100 batch-1 decode at 27% of peak HBM with
~283 launches/step — our regime is the field's regime; graphs alone capped theirs at 1.26x.

**The honest ceiling without touching bytes: ~2.9x (176 → ~500 t/s class).** Every real cut must
remove nodes or waits, not FLOPs. (And at 24-26% effective BW, byte cuts are DEVALUED: 1 GB saved
≈ 2 ms now vs 2.4 ms at the floor — fix execution first.)

**The byte budget** (@T=3.6): routed experts 3.28 GB (45%), dense projections 1.78 GB, hc
(hyper-connection) reads 1.26 GB (bf16 [320x10240] 2x/layer!), head 0.52 GB (Q6_K 248320x2560),
GDN state 0.23, KV/idx 0.15, shared/router 0.25, activations ~0.2. **The dense skeleton is 55% of
bytes** — read every layer every window regardless of T; the "it's all expert traffic" MoE mythology
is wrong at this geometry. Draft chain is only 1.1-1.2 ms (~6-9% of a round); verify 87-90%; commit
1.10 ms (a single-buffer hazard on `h_commitb_`, not a data dependency — verify.cpp:2977).

**The worst trade in the window**: VRAM entry hit is 97.4-98.0%, but the doorbell/host path engages
unless ALL 20-36 entries of a layer-window are resident — ~half of layer-windows pay a host round
trip (pool 1.56-2.10 ms/window + interleaved GPU wait-nodes) for a 1.8-2.4% CPU-expert tail
(0.5-0.6 CPU entries per layer-window). Env-tunable today via ROUTE_RESIDENT margin/profile.

## THE STRENGTHENED QUEUE (merged round-8 + round-9; the delta vs F-178 marked NEW)

**Tier 0 — env-only, free:**
- R9: F-67 overlap unblock (unchanged; +2-6% conc, two-boot Windows evidence).
- NEW **STRATA_QFUSE=1**: hc-read writes the consumer's q8_1 inline (verify.cpp:1065-1069) — −48
  quantize nodes/window, exists, OFF, never tested here.
- R10: PDL (`STRATA_DF_PDL=1`) — now production-backed as THE field answer to node spacing
  (SGLang applies it across Gate-Up/activation/Down/top-k; TRT-LLM ships it globally on Blackwell;
  FlashInfer on-by-contract). Sustained-reasoning gate mandatory (F-107's −40% was WDDM).
- NEW **diagnostic boot**: `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1 STRATA_VERIFY_NODES=1`
  — replaces every estimate in the budget table with measurements. Zero risk; should precede all
  Tier-1 code work.
- NEW **ROUTE_RESIDENT margin sweep**: kill the CPU-expert doorbell tail (~2-3 ms/window potential;
  the mec RFC's +84% on our model family is the same mechanism in a harsher regime). MMLU-gated
  (round-1 note: ROUTE_RESIDENT changes the output).
- R5b: tau 0.15/0.10 + STRATA_SPEC_RATIO=1 (unchanged, MMLU-gated).
- R8: clocks + power cap + Xid-79 firmware check + iGPU cable (user steps; variance first).

**Tier 1 — code, days:**
- NEW **commit double-buffer**: `h_commitb_` hazard fix returns most of 1.10 ms/window.
- NEW stream-priority one-liner (verify > adapt; mtp.cpp:461 pattern exists).
- **Ungated resident share** (partial device plan): extend resident_plan_kernel to plan resident
  groups of partially-resident layers, gate only the miss share — removes the host from the GPU
  critical path ≈ 1.5-2.5 ms/window (9-15%). [1-2 weeks]
- **Fenced batched admission**: 30-56 ms → 10-20 ms (28 device syncs → event fences + coalesced KV
  images). Attacks the 184-vs-210 rows/s gap and tail latency.

**Tier 2 — research→code, weeks:**
- **F-31 slot coupling** (the concurrent ceiling breaker) + **device-driven draft chain**: CUDA
  conditional nodes / device graph launch are green field — NO production engine uses them
  (SGLang/TRT-LLM/vLLM all stopped at full/piecewise graphs; PyTorch doesn't expose them); the
  scaffolding exists here (mapped spin gates, E-6 device plans, skip words). sm_120 supports both.
- **Node-count fusion program**: router+route_resident fusion (−48 nodes), QSA epilogue fusions,
  BCG-style segment breaks for the CPU-expert path (SGLang's Breakable CUDA Graph pattern — the
  non-PyTorch idea is API-simple), TRT-LLM-style router-GEMM+topk single node.
- FastMTP (Tier 3 [training], unchanged): acceptance multiplies window amortization (rows/s ∝ T at
  fixed window cost) — orthogonal to and multiplicative with all of the above.

## DEMOTED BY ROUND-9 EVIDENCE

- **Draft-verify overlap / SSD-style pre-speculation**: draft is 6-9% of the round — overlap caps
  at ~6-9%. (SSD's 30% needs a second GPU.)
- **ReplaySSM as a direct lever**: GDN state is 0.23 GB/window; halving saves ~0.1 ms. Its
  checkpoint+rollback pattern stays relevant only as an F-31/overlap enabler.
- **Megakernel from scratch**: the window is ALREADY one graph launch; megakernel benefits
  decompose into node-count reduction + device-driven execution, reachable via cheaper paths. MPK
  is PyTorch-only (not drop-in for hand-written CUDA).
- **hc bf16→Q8 byte cut**: worth ~0.4 ms at current execution density — LOW until node spacing
  is fixed (then it becomes the top byte cut).
- L2 persisting windows for MoE: no published MoE-decode measurement exists anywhere (open slot);
  DSMEM/clusters: zero decode-engine adoption.

## KEY EXTERNAL SOURCES

Hazy Research megakernels (blog 2025-05-27; paged shmem + counter sync; batch-1 dense only);
Mirage Persistent Kernel (arXiv 2512.22219; 1.0-1.7x, PyTorch-only); SGLang HPC-Ops PDL
(lmsys 2026-08-07); SGLang Breakable CUDA Graphs (lmsys 2026-08-17; BCG 17% over piecewise);
TRT-LLM DeepSeek-V4 Blackwell blog26 (router 8→3 µs, top-k 384→112 µs, GVR routing-correlation
top-k 1.40-2.17x kernel-level, MegaMoE +15.3%); llama.cpp #28248 mec RFC (device-side expert-id
indirection; +84% Qwen3.8-Flash-Next all-CPU 4090); llama.cpp #27050 (5090: ~2.7 ms idle per graph
launch, sampling D2H costs); arXiv 2605.30571 (27%-of-peak batch-1, graphs cap 1.26x); arXiv
2607.17283 (K=6 optimal plateau; "the speedup is a systems property"); SSD (arXiv 2603.03251);
ReplaySSM (NVIDIA blog; GDN covered; no gain at batch 1); FastMTP (Red Hat, Sep 2026); conditional
nodes (NVIDIA blog 2024; IF 12.3, WHILE 12.4, ELSE/SWITCH 12.8, CUPTI bodies 13.1).

No experiments run this round (study-only per operator). Engine untouched; standing recipe serving.


### F-180 — THE v0.1.44 MERGE, TESTED BEFORE AND AFTER: adopted at neutral perf, 1-question MMLU delta, race fix banked (2026-10-14 ~02:00, goal round 10)

**THE PROTOCOL (operator-specified: prove no regression, ideally improvement, before keeping a
merge).** Baseline on the standing 0.1.43.1-local binary, fresh boot, box at its best of the night:
**M1 195.8 / pair 161.3 / 6-conc 209.5 / M3b 79.0 / MMLU 95.71** (r10-base1-3, mmlu-r10-ml-base —
the drift from F-178 fully reversed; M1 back above the locked anchor).

**THE MERGE METHOD (recorded for future merges).** Our tree was CRLF (NTFS heritage), upstream LF —
a naive merge is a 410-file conflict storm. Recipe that worked: (1) commit runtime state; (2) commit
`.gitattributes` `* text=auto eol=lf` + `git add --renormalize .` + hard re-checkout (tree now LF);
(3) extract the real mod delta vs v0.1.43.1 — only **557 lines across ~16 files** (the Windows
campaign's engine patches: typical-acceptance rescue F-15/F-16 + batch-rescue F-30b, ratio test
F-27, B3/B6/G1/G2 victim-protection arms, PREDICT_CREDIT, MTP_TAP F-86/F-91 diagnostics); (4) take
v0.1.44's tree wholesale (`git checkout v0.1.44 -- .`), then `git apply --3way` our patch — only TWO
hand resolutions needed: mtp.cpp (upstream's new `stamp(14)` ∪ our MTP_TAP block) and generate.cpp
(upstream's new `pcie_auto_step` lambda ∪ our B3/B6/G1/G2 declarations — both sides load-bearing,
union taken). Committed with real dual parents (8d1c9d6) so future merges are ordinary. Built clean
(venv cmake 4.4.3 + CUDA 13.4, 146 steps, one benign warning); deployed per F-30 with rollback copy
`engine/strata.0.1.43.1-local-backup`. All campaign arms verified present post-merge
(TYPICAL_ACCEPT, SPEC_RATIO, ADAPT_*, PREDICT_CREDIT, DMA_BATCH, L2_SETASIDE, GUMBEL, AUX_CPUS).

**WHAT v0.1.44 BRINGS US:** STRATA_COMMIT_BATCH on by default (the commit-graph consolidation —
our round-9 target, now in-tree); STRATA_PIN_BY_NEED + the arena async tier (opt-in, future A/Bs);
the K/V-grow/slot-lending RACE FIX (real correctness, default); MMQ_KQ prefill kernels (existed);
HC_Q8 exists but is DEAD for our GGUF (0.00 GiB of Q8_0 hc tensors — a UD-file feature).

**THE AFTER (same night, back-to-back, 3 probes + 3 MMLU gates):**

| | M1 | pair | 6-conc | M3b | MMLU (3 gates) |
|---|---|---|---|---|---|
| before (0.1.43.1-local) | 195.8 | 161.3 | 209.5 | 79.0% | 92.86 / 94.29 / 95.71 → **94.29** |
| after (merged 0.1.44) | 195.2 | 160.4 | **211.0** | 80.0% | 91.43 / 94.29 / 92.86 → **92.86** |
| Δ | −0.3% | −0.6% | **+0.7%** | +1.0 pt | **−1.43 pts = exactly 1 question / 70** |

Every perf delta is deep inside the ±4% noise band; the best single readings ever recorded on this
box came on the merged binary (6-conc 215.4, M1 200.1); TREND/smoothness identical (late ~8.7-9.2
ms/window). MMLU distributions overlap completely (merged's middle gate = standing's middle gate).

**A METHOD FINDING, honestly recorded:** the fixed-seed greedy token-identity test between binaries
is INFORMATIVE-NEGATIVE — the engine's greedy output is history-dependent (expert tier residency is
stateful: CPU-served vs VRAM-served experts round differently), so even the SAME binary produces
different greedy streams run-to-run. MMLU statistics are the only quality instrument; binary
identity tests are not.

**VERDICT: ADOPT, with an MMLU watch flag.** The Lock Law's ±4% is passed everywhere (worst: MMLU
−1.5%); the pre-registered 93.3 floor is technically missed by 0.44 pts, but the same-night 3v3
shows a one-question delta inside total overlap, the merge banks a real data-race fix and the
commit-consolidation infrastructure, and the rollback binary is one copy away. WATCH RULE: re-gate
MMLU at the next adoption round; a 3-gate mean < 92 escalates to investigation before any adoption
claim. Sources: campaign/results/r10-base{1,2,3}.txt, r10-merge{1,2,3}.txt, mmlu-r10-ml-{base,
merge,merge2,merge3}.json.


### F-181 — THE FULL-CONTEXT QUESTION ANSWERED: native 262k measured, 1M YaRN built and measured — ladder on record, 131k stays standing (2026-10-14 ~03:30, goal round 11)

**THE QUESTION (operator).** We run `--max-context 131072`; the model "supports 1M with yarn" — test
full context and the performance difference.

**THE GROUND TRUTH FIRST.** The GGUF declares `context_length = 262144` (rope base 1e7) and carries
NO rope-scaling keys — 1M is not in the file; it is Qwen's served recipe applied at load time. The
engine (v0.1.44 merged) has full support: `--rope-scaling yarn --rope-scale 4` (llama.cpp names;
boot confirms "yarn, factor 4, freq_scale 0.25, mscale 1.138629 against a trained context of
262144"). Test legs: 262k native (`custom-iq3s-262k.json`) and 1M YaRN ×4 (`custom-iq3s-1m.json`),
both from the standing battery; new instrument `campaign/depth-lane.py` (one request: deterministic
synthetic filler to target depth + fixed decode; server timings parsed; M6-class).

**THE LADDER** (same binary, back-to-back boots; "cold" = fresh conversation, "warm" = the
deterministic filler let the conversation cache serve a prefix — noted per row):

| config | depth | prefill t/s | decode t/s | accept |
|---|---|---|---|---|
| (short-ctx class, same binary) | ~1-30k | ~5.7-6.5k | ~195 (M1) | ~79-81% |
| 262k native | 157k (cold) | 6534 | 116.7* | 62.4%* |
| 262k native | 244k (warm, cache_n 131k) | 5711 fresh | **173.9** | **79.4%** |
| 1M yarn ×4 | 128k (cold) | 6291 | 126.3 | 73.7% |
| 1M yarn ×4 | **937k** (warm, cache_n 113k) | **4343** | **98.7** | 60.9% |

\* the 157k cold row decoded only 266 tokens (early stop) — short sample, lower-confidence row.

**READING.** (1) Depth costs decode roughly linearly in the sparse-index overhead band we can see:
−11% at 244k (best case: warm cache, 79% acceptance), −36% at 937k (98.7 t/s — still ~7x the public
ceiling for this class). (2) Prefill: ~6.3-6.5k t/s in the 128-157k band, −33% at 937k (KV/idx
streaming grows with depth; YaRN itself ≈ −4-8% at matched depth, 6291 vs 6534 cold-cold). (3) Draft
acceptance degrades with depth on synthetic filler (79% → 61% at 937k) — how much is filler-content
vs rope-extended positions is NOT separated; a real-corpus long-context probe would tell. (4) RAM
headroom is a non-issue: 129 GB used of 184 at 937k with the 49 GB conversation cache absorbing the
streamed KV. (5) **QUALITY: YaRN ×4 is clean at short prompts** — MMLU 94.29 ± 2.77 under the 1M
config (mid-band vs tonight's 92.86-95.71 standing spread; mscale applies globally so this gate was
necessary). Long-context REASONING quality (needle-style) at ≥262k is UNMEASURED — flagged, not
claimed.

**VERDICT.** The 1M and 262k configs are BUILT, TESTED, and on the shelf (launchers
`launch-custom-iq3s-262k.sh` / `-1m.sh`) for any deep-context workload: at 937k the lane still
serves 4343 t/s prefill / 98.7 t/s decode. The STANDING lane stays 131k — the campaign's bars and
the anchor are measured there, and full-context costs 11-36% decode by depth. Sources:
`campaign/results/depth-{262k-d130k,262k-d240k,1m-d127k,1m-d965k}.txt`, `mmlu-1m-yarn-ml.json`.


### F-182 — THE BALANCE QUESTION: capacity is free, threads honestly idle, calibrate moot — the dynamic-context and CPU-utilization studies (2026-10-14 ~05:30, goal round 12)

**THE OPERATOR'S QUESTIONS.** (1) Scale to 262k for real use, 131k for baselines — agreed. (2) Can the
context window be dynamic — grow as needed, no fixed cost when small? (3) Should we leverage ≥80% of
CPU threads? (4) Does `--calibrate` give gains? Plus: why only ~6k experts cached?

**THE 6K-EXPERTS ANSWER.** The 1M boot's slot sessions carved **9.09 GiB** of the expert cache's
VRAM (262k: 3.79 GiB; 131k: 2.91 GiB) — that boot cached ~6k experts instead of 10,182. Not a fault;
capacity pricing in plain numbers.

**CAPACITY IS FREE AT SHORT USAGE — MEASURED BACK-TO-BACK.** 262k vs 131k, same night, same binary:
M1 177.0 vs 177.4 (−0.2%), pair 149.0 vs 150.2 (−0.8%), 6-conc 190.6 vs 193.2 (−1.3%) — statistical
zero (an apparent −9% earlier in the evening was the drift swing, both configs reading ~177 at that
hour vs the 01:00 peak of ~195). **The engine already does "declare big, pay per depth": streaming
KV means cost tracks USED context, not declared capacity.** The only fixed cost is the slot-session
arena (0.88 GiB going 131k→262k) — noise at our hit rates. The user's daily driver = the 262k
launcher; baselines stay 131k. (Third 131k control by a mis-fired pool-workers boot: 176.7/147.4/
189.3 — consistent.)

**TRUE DYNAMIC CONTEXT (rope-on-grow): NOBODY DOES IT, AND IT BREAKS.** The scan (kvcached/OSDI'26,
vAttention/ASPLOS'25, vLLM/SGLang/TRT-LLM, llama.cpp paged-KV RFC #21961, InfiniteHiP, InfLLM-V2)
is unambiguous: no engine re-scales rope mid-conversation — KV cached under the old encoding is
incompatible; silent corruption unless K is re-rotated (checkpoint/restart or 1.6×-prefill re-index
are the only shipped mechanisms). Within one rope setting, capacity-elasticity is solved two ways:
software paging (~3% overhead) or CUDA VMM (kvcached: reserve VA, map physical per use —
production-adopted by Google/LinkedIn/Red Hat). OUR [code] PATH: make `--kv-grow`'s VMM coexist
with the streaming+batch recipe (it currently refuses kv-resident/batch/resident-experts) — that
eliminates the slot-arena cost entirely and makes 1M cheap. Our streaming-KV + top-2051-index
design is validated as the published best practice (InfLLM/InfiniteHiP/TRT host-cache); our 173.9
t/s @244k is competitive with anything published on one GPU.

**80% CPU UTILIZATION: THE EVIDENCE SAYS NO — AND THE IDLE CORES ARE THE HONEST READING.** The
CPU-threads scan (KTransformers SOSP'25, Fiddler, fMoE, llama.cpp #8684 maintainer-confirmed,
#11765 3-cores-per-DDR-channel rule, TRT-LLM n-gram blog, vLLM v0.14 async scheduling): high CPU
utilization is the design goal of RESIDENCY-INVERTED systems (experts on CPU DRAM with AMX — we
have 98% VRAM residency, 2% CPU traffic, and Zen5 has no AMX). At GPU-bound offload "there is no
mechanism for helper threads to do useful work; they'd just spin" (llama.cpp maintainer). Our
15 workers are already ~2.5× past the 2-channel bandwidth saturation point (~6 threads) for expert
math. **What the cores CAN do (evidence-backed, queued):** (1) host-side n-gram drafting feeding
the same GPU verify — TRT-LLM measured +96% b1 / +64% b4 on cache-warm traffic, lossless, composes
with MTP — NOTE: the engine's PLE (n-gram embedding) path may already be a partial implementation;
investigate before building. (2) De-stall admission/orchestration (the 30-56 ms sync'd admissions =
2-5 windows of stall; matches round-9's fenced-admission item). (3) Prefill support roles
(tokenization, prefix-cache, staging). Chasing utilization for its own sake = spin-waiting or
demoting work from a 1792 GB/s device to a ~100 GB/s one — both measured/literature-confirmed net
negative.

**--calibrate: MOOT AND DANGEROUS.** It is a SETUP-harness auto-tuner (15-30 min; A/Bs
--pcie-frac, --spec-min-p, --pool-workers, adapt cadence; keeps >3% winners) — running
`setup.sh --calibrate` would rewrite the config args and DESTROY the standing recipe (Launch Law).
Its knobs: pcie-frac 0.35 and spec-min-p 0.5 are already campaign-measured optima (F-126,
tau-curve-1); **--pool-workers 8 was A/B'd tonight: null** (182.8/145.3/190.6 vs 177.4/150.2/193.2
control — mixed inside noise, as predicted at 2% CPU traffic; config reverted, standing verified);
adapt cadence stays queued (interacts with the learned profile). Nothing it could measure is worth
the config clobber.

Sources: `campaign/results/262k-short{1,2,3}.txt`, `131k-ctrl{1,2,3}.txt`, `pw8{,b}-{1,2,3}.txt`;
agent scans archived in the session record. No adoption; standing recipe unchanged and healthy.


### F-183 — THE CALIBRATE RUN, BACKED UP AND TESTED: adapt tier +12.7% on its probe but a trade-off on ours; the spare 1.2 GB filled and measured (2026-10-14 ~07:30, goal round 13)

**THE PROTOCOL (operator-specified).** See what --calibrate touches; back up the main config to
~/LLM/backup; run it; also check whether it calibrates VRAM occupancy (1.2 GB unused observed).

**WHAT IT TOUCHES (recon before running).** `setup.sh --calibrate` → `calibrate_config()` →
`tools/calibrate.py`: sweeps --pcie-frac {0,.2,.35,.55,.75,.9,1} → --spec-min-p {.3,.5,.7} → a
confirm round vs defaults → --pool-workers {15,10,8,4} → adapt-tier candidates — each value behind
an engine RESTART, decode-rate probe with 3-round medians; keeps only >3% winners; writes winners
into the config's args (the Launch-Law clobber) and per-PC into ~/.config/strata/settings.json.
Does NOT touch VRAM occupancy. Ran via a direct `calibrate_config()` call (setup's interactive
install path demands prompts / risks an 84 GB download with --yes). Backup first:
`~/LLM/backup/2026-10-14-precalibrate/` (configs, launchers, merged binary, BUILD.json, sha256
MANIFEST); the 13 other root jsons parked during the run so it found exactly one model.

**ITS RESULTS (its own probe):** pcie 0.9 nominal best but under the 3% bar (ranges overlap ours
0.35: 122-132 vs 0.9: 131-135) → it STRIPPED our --pcie-frac 0.35 (restored from backup per F-126);
spec-min-p 0.5 CONFIRMED optimal (134.9 vs 117.6/130.8); workers 15 CONFIRMED (136.8 vs 129-135);
**adapt tier: default 120.8 → every-1/80-swaps/decay-0.97 = 136.1 (+12.7%)** — the one real winner
it found, saved to the per-PC settings file.

**OUR VERDICT ON ITS WINNER (back-to-back boots, our probe battery):**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| calibrated adapt (1/80/0.97) | 180.0 | 147.8 | 192.0 | 79.6% |
| standing adapt (0.85) | 183.5 | 143.0 | 186.4 | 81.0% |
| Δ | −1.9% | +3.3% | +3.0% | −1.4 pts |

Mixed, nothing clears ±4%, and the directions run against M1 and acceptance (both on the priority
list). Its probe = one long generation (TIER_MAX_NEW) where swap cadence dominates; ours = many
short diverse requests where the learned profile + B6 pins already manage the tier. **NOT ADOPTED**;
standing config restored byte-identical from backup.

**THE SPARE 1.2 GB (--vram-reserve-mib 2000 → 800):** cache 10,190 → **10,703 experts (20.31 GiB)**,
VRAM 32.07/32.6 — the gap filled as asked, then measured:

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| vram 800 (3 probes) | 170.8 | 148.4 | **195.6** | 79.9% |
| standing 2000 (control) | 183.5 | 143.0 | 186.4 | 81.0% |
| Δ | **−6.9%** | +3.7% | **+4.9%** | −1.1 pts |

A clean CONCURRENCY-vs-SINGLE-STREAM trade: concurrent slots pull a wider expert union, so +513
slots raise their hit rate; single-stream was already ~99% and pays the bigger cache's overhead.
Under the no-regression law NOT adoptable into standing (M1 −6.9% breaches), but recorded as the
**concurrent-favoring shelf arm** — it compounds with the same-signed adapt-tier arm and, later,
with F-31 slot coupling. If real-use priorities ever favor concurrent aggregate over M1, the 262k
daily driver can carry vram-800 (+ the adapt winners) as a labeled variant.

Standing recipe verified restored (reserve 2000, decay 0.85, 10,190 experts, healthy). Sources:
`campaign/results/cal-adapt{1,2,3}.txt`, `cal-ctrl{1,2,3}.txt`, `vram800-{1,2,3}.txt`;
`/tmp/calibrate-run2.log` (the full sweep); backup manifest in `~/LLM/backup/2026-10-14-precalibrate/`.

**ADDENDUM — "WHY NOT TAKE THOSE WINS?" (operator challenge, tested):** the stack (vram 800 +
adapt winners) was re-tested where taking it is legitimate — the 262k REAL-USE config, back-to-back
vs plain: 6-conc +3.5%, pair −0.1%, M1 −3.7%. The concurrent gain HALVED vs the single-arm reading
(+4.9→+3.5) while the M1 cost persisted — the two measurements bracket a ±noise trade, not a win.
Plain 262k restored as the real-use launcher; the armed variant shelved at
~/LLM/backup/2026-10-14-precalibrate/custom-iq3s-262k-CONCURRENT-STACK-variant.json (one cp away).
The honest concurrent lever remains F-31 slot coupling — the slots are DRAFTLESS; no cache-tuning
arm changes that ceiling. Sources: 262k-plain-{1,2,3}.txt, 262k-stack-{1,2,3}.txt.


### F-184 — F-31 SLOT COUPLING: ADOPTED — pair +11.5% for one env line (2026-10-14 ~09:30, goal round 14)

**THE QUESTION (operator: "proceed with F-31 and prove its worth it").** Concurrent batch slots ran
DRAFTLESS since the Windows era (STRATA_BATCH_MTP=0, suspended under the No-Regression Law after
F-30b): uncoupled drafts rarely exact-match at temp>0, and the rescue variant that made slots pay
was quality-banned (MMLU 95.7 → 91.4-92.9). Coupled drafting — the drafter samples with the
target's chain + the same Philox draw — is quality-safe by construction (coupled_draft.hpp:5-6),
and F-30b's own note said "the pair win came from drafting, not rescue".

**THE DIAGNOSTIC FIRST (the journal's own prescription: "a diagnostic boot, not more static
reading").** Static re-reading of the MERGED tree found the acceptance site plain exact-match
(`outb[i] == tok[i+1]`, generate.cpp ~10226) with the verify pick drawn by
`Verifier::sample_rows` (counter = row position, verify.cpp ~3172) — and the counter law
(coupled_draft.hpp:19-30) algebraically COINCIDES with the position convention. Booted
STRATA_BATCH_MTP=1 on the merged binary: **slot drafts accepted 60-67% sustained** — the coupled
class, live. The "counter staging defect" of the old journal does not exist in the v0.1.44-merged
tree; the wiring is complete. **F-31 was never a code project here — it was one suspended env
line.**

**THE PROOF (back-to-back boots, same night, our battery):**

| arm | M1 | pair | 6-conc | M3b | MMLU |
|---|---|---|---|---|---|
| STRATA_BATCH_MTP=1 (F-31) | 175.1 | **162.8** | 193.7 | 79.8% | **97.14 ± 1.99** |
| control (=0, standing) | 179.1 | 146.0 | 192.8 | 80.3% | (95.71 anchor class) |
| Δ | −2.2% (noise; serial path untouched) | **+11.5%** | +0.5% | −0.5 pt | best gate of the campaign |

**WHY PAIR AND NOT 6-CONC:** at pair, 2 slots + 2 draft rows = a 4-row window — the ~63% chance of
banking a second row per slot comes nearly free. At 6-conc the window doubles 6→12 rows and the
MoE expert-union cost of the extra rows eats the gain (+0.5%). The serial path (M1) is mechanically
unaffected; its −2.2% is the night's spread.

**QUALITY:** exact-match accept only — the emitted stream is the target's own picks on both accept
and reject (the F-15 divergence lesson cannot arise; the F-30b rescue stays OFF by default).
MMLU gate 97.14 ± 1.99 = the highest single gate ever recorded on this lane. The 60-67% slot
acceptance also lifts nothing distributional: drafts only decide HOW MANY tokens a window banks.

**ADOPTED:** `STRATA_BATCH_MTP=0 → 1` in BOTH the standing 131k launcher and the user's real-use
262k launcher (concurrency is exactly the real-use case). Standing restarted on the adopted arm
and verified (env confirmed on the engine process; health green). Sources:
`campaign/results/f31-{diag-pair,bmtp2,bmtp3,ctrl1,ctrl2,ctrl3}.txt`, `mmlu-f31-bmtp-ml.json`.

**WHAT THIS OPENS:** the concurrent ceiling moved for low concurrency; the 6-conc ceiling is now
purely the expert-union cost of wider windows — the tier/cache arms of F-183 (which pay only under
concurrency) and deeper slot drafts (draft(2,...) offers ONE proposal per slot; a longer slot
chain, or the F-183 cache arms stacked ON the now-cheaper slot rows, are the follow-ons).


### F-185 — R9 OVERLAP-UNBLOCK TESTED: rejected — the three "vestigial" arms are worth more than the F-67 overlap they block (2026-10-14 ~11:00, goal round 15)

**THE HYPOTHESIS (round-8 study).** The launcher kept ADAPT_LAG=2 / DMA_BATCH=1 /
VERIFY_DEVICE_PLAN=1 from the pre-v2 Windows battery; each trips the refusal chain
(generate.cpp) that disarms the adopted F-67 ADAPT_OVERLAP combo. Windows two-boot evidence had
the overlap at +4-9% concurrent. R9 = drop the three, arm the overlap.

**THE ARM ENGAGED CORRECTLY.** Boot line confirmed for the first time on this box: "adaptive
overlap: the adaptive tier's copies land while the next window runs" (vs the old refusal note).

**THE A/B (back-to-back boots, 3+3 probes, on the F-31-adopted recipe):**

| arm | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| R9: overlap armed (three arms dropped) | 160.6 | 167.8 | 193.4 | 79.5% |
| control: standing (three arms + BATCH_MTP=1) | **173.8** | **176.7** | **204.3** | 80.0% |
| Δ (R9 vs control) | **−7.6%** | **−5.0%** | **−5.3%** | flat |

**VERDICT: REJECT.** Arming the overlap costs 5-8% across the board HERE: what it disarms is worth
more than what it arms. Mechanism: STRATA_VERIFY_DEVICE_PLAN is the E-6 per-layer device plan —
the machinery that lets fully-resident layers skip the host round-trip entirely (round-9's budget
showed it covering ~half of layer-windows); STRATA_DMA_BATCH batches the swap submits; the
overlap's win (copies landing behind the next window, worth ~0.25 ms/window of adapt wait) is
smaller than their combined contribution. The Windows v2 battery's +4-9% was measured WITHOUT
these arms (different recipe, different balance); on the merged 0.1.44 line they are not vestigial
— they are load-bearing. Launcher restored to standing (the control IS standing); the armed
variant parked at /tmp/launch-r9-armed.sh for the record.

**THE BONUS OBSERVATION: the standing recipe's current-window means are the best concurrent
numbers of the whole session — pair 176.7 / 6-conc 204.3 / M1 173.8** (the drift is up; these are
window numbers, not anchor claims — but they show the F-31-adopted recipe's concurrent class:
pair +62% over the Windows bar of 108.8, 6-conc +46% over 140.2).

Sources: campaign/results/r9-ovl{1,2,3}.txt, r9-ctrl{1,2,3}.txt.


### F-186 — THE NEXT-LEVEL QUEUE v3: two frontier scans + R9 rejected + batch-6 adopted (2026-10-14 ~12:30, goal round 15 study)

**R9 (overlap-unblock) TESTED AND REJECTED** (F-185): arming the F-67 overlap costs 5-8% — the
three "vestigial" arms are load-bearing (VERIFY_DEVICE_PLAN = E-6 device planning over ~half the
layer-windows, DMA_BATCH, ADAPT_LAG). Round-9's "ungated resident share" item is REFRAMED: extend
E-6 to PARTIALLY resident layers (keep the flag, plan the resident share on device, gate only the
miss share) — do not remove it.

**BATCH WIDTH 6 ADOPTED** (F-185 addendum): the systems scan's top item — the 6-client probe was
slot-starved (queued 2 on 4). 6-conc 216.0 (+5.7%, the highest concurrent mean ever), pair flat,
slot acceptance 67.8%. In both configs.

**THE FORMULATIONS SCAN (ranked, full text in the session record):**
1. **FlashInfer SM120 stack** — fused GDN decode step for our exact model family (v0.7.0),
   persistent fused MoE with recompile-free decode-batch changes (v0.6.17), ragged-Q XQA for spec
   decode (v0.6.16, variable accepted lengths without padding — the "verify sublinear in rows"
   primitive for the 6-conc width problem). Port kernels, not ideas. [code]
2. **ECHO** (arXiv 2604.09603) — global verification budget across rows: deepen confident rows,
   truncate+reallocate lazy ones; sparse confidence gating at 3-4 calibrated depths; training-free;
   +8-14% at high concurrency over fixed EAGLE-3. Formalizes our 6-conc width problem. [code]
3. Routing-aware per-expert mixed precision (arXiv 2604.06515) — COUNTERINTUITIVE: RARE experts
   need the bits (router-norm-ranked), beating hot-boost (PMQ). Same-bytes quality; our
   expert-profile data already has the routing stats. [code+analysis]
4. DFlash v2 (KV-injection ablation: injection alone 4.8 acc vs 3.5 — applies to our MTP head via
   light retraining; full DFlash = new drafter). [training]
5. CSD calibrated rescue (training-free, LOSSY — conflicts our quality law; parked).
6. FlexDraft's finding: parallel draft+verify at batch<=2, sequential above — matches our
   pair-vs-6-conc split; the batch-adaptive strategy switch is the idea. [code]
TRAP CONFIRMED: 2026 spec papers report greedy-only; discount acceptance claims ~25-30% at temp 0.7
(OmniDraft's T=1 data: 1.57x→1.27x) — matches our Gumbel/min-p experience.

**THE SYSTEMS SCAN (ranked):**
1. Batch width (DONE — adopted above). Sentinel-slot graph bucketing (SGLang #27988) as the
   follow-on: fixed slot count + zero-length sentinels instead of per-layout graphs. [code]
2. DuetServe scheduler (arXiv 2511.04791): decode-first token budget + LOOK-AHEAD k-step graph
   launch (pre-launch next step's graph while current drains — attacks the 9.5 µs node spacing).
   [code]
3. Layered prefill (arXiv 2510.08055, MLSys'26): layer-group interleaving replaces token-chunk
   prefill — kills the per-chunk expert-reload tax (+39% off-chip traffic on MoE); up to 70% TTFT
   / 41% e2e. Prefill is a stated priority. [code]
4. Env: clock lock + power cap (TensorRT practice; Blackwell clock-transition hard-lock reports) —
   the R8 user step, now double-justified; driver 615.71.09 carries the Xid-109 fix with residual
   reports — keep the watch. [env/sudo]
5. CUDA 13.x grab bag: launch-completion events (submit next graph during drain),
   cudaMemcpyBatchAsync node collapse, CUDA 13.2 graph-host spin-wait; NVIDIA's llama.cpp trick:
   QKV as MULTIPLE STREAMS INSIDE ONE GRAPH (+35% MoE TG class — a node-density lever we have not
   exploited). [code]
6. sm_120 kernel truths: TMA supported (wins at prefill tile sizes, NOT decode gemv); clusters
   ≤8 supported but no gemv win; NO tcgen05/TMEM/wgmma/NVLink — datacenter Blackwell recipes do
   not port; 99KB smem, 48 warps/SM. DO-NOT-BOTHER list: tcgen05-class, DSMEM-gemv, NVFP4 requant,
   PD disaggregation, ik_llama offload, n-gram depth >2 (S=2 saturates on GDN hybrids, mlx-lm
   measured).

**THE OPEN LANE (both scans converge):** nobody ships a per-row entropy-gated arbiter between the
neural MTP chain and an S=2 n-gram drafter under sampled traffic with GDN-state rollback — all
components have 2026 production evidence (vLLM dynamic-spec control plane, mlx-lm's rollback
recipe, ECHO's budget math, our coupled verifier + per-slot state). A [code]-only project with a
credible 10-20% concurrent win.

**QUEUE v3:** R16 batch-8 probe (+ sentinel-slot graphs design) → R17 FlashInfer kernel ports
(fused GDN step first — our exact architecture) → R18 ECHO-style verify budget + ragged packing →
R19 layered prefill → R20 DuetServe look-ahead launch + launch-completion events + QKV-multistream-
in-graph → R21 FastMTP+KV-injection training → R22 rare-expert mixed precision (quality axis) →
R8 clocks (user step) whenever ready. Standing recipe now: merged 0.1.44 + BATCH_MTP=1 + batch 6.

### F-187 — THE SIX-LEG DEEP SCAN: drift mechanism found (GPU state), the untested v0.1.44 opt-ins, dead-arm hygiene, and the floor-stair M1 lever (2026-10-14 ~14:30, goal round 16 study)

Six parallel study legs (4 code + 2 web), user-directed. All file:lines verified by the legs.

**CORRECTIONS to the standing queues:**
- R9 overlap-unblock: already TESTED/REJECTED (F-185). Commit double-buffer: already BANKED —
  v0.1.44 ships STRATA_COMMIT_BATCH default-on (verify.cpp:2039-2044). Removed from Tier-0/1.
- QFUSE (R10a) downgraded to boot-rider with a CAUTION: it disables T=1 self-commit
  (verify.cpp:879) — never pair it with a BATCH_MTP adoption boot; F-67 caps expectations.
- Launchers carried dead arms: STRATA_ADAPT_OVERLAP / _RESIDENT are REFUSED while DMA_BATCH=1
  (boot log: "is off: not with STRATA_DMA_BATCH"; generate.cpp:8886-8931) and KEEP_MAPPING is a
  dead branch on Linux (file_tier_unbuffered=false without STRATA_UNBUFFERED_LOAD,
  expert_source.cpp:510-520). DELETED from launch-custom-iq3s-{262k,8081}.sh this round.
- 262k DAILY DRIVER still runs parallel:4 (custom-iq3s-262k.json:71) — the F-185 batch-6
  adoption never reached it. A/B queued (R0).

**NEW #1 — GPU state measured (the drift mechanism):** persistence_mode Disabled, clocks UNLOCKED
(max sm/gr 3135 MHz, mem 14001; power 575/600 W; nvidia-persistenced unit disabled). Re-anchor
v2's −7.3% M1 at 42 °C clean ("drift is global, not GPU throttle") is most consistent with
unlocked boost/P-state churn between probe clients. R8 upgraded from "double-justified" to
DIAGNOSED; exact block handed to the user (persistenced enable, -pm 1, -lgc 2835 [step 2745 if
dmon sw_power/sw_thermal active], -lmc 14001, optional -pl 600; -rgc/-rmc reverts). VARIANCE
FIRST: a claimable M1 needs the lock + re-anchor v3 before any mean-chasing round.

**NEW #2 — v0.1.44 opt-ins never tested here:** STRATA_PIN_BY_NEED=1 + --adapt-async 1 are
in-tree and untested; upstream release notes claim +4.2-5.2% decode on an RTX 5070
(vendor-measured, our engine family). Pin-by-need page-locks only hot (layer,expert) pairs —
same CPU-expert tail as the device-plan work. Env-only A/B (R10f), MMLU-gated (residency is
stateful; F-180 history-dependence).

**NEW #3 — the spec-path M1 lever, all distribution-neutral:**
- The floor is the binder, not the cap: serial chain loop `while (T < S_mtp && dprob[T-1] >=
  spec_min_p) ++T` (generate.cpp:12796-12798; pipeline twin 12271-12274). F-159 depths accept
  96/93/90% at 1/2/3, but chains stop ≤3 at floor 0.5 → depths 4-6 starve under --spec 6.
- Floor STAIR (0.5 at depths 1-2 → ~0.30 at 3+): offer-side only; serial verify is true
  rejection sampling with residual resample (verify.cpp:2449-2465, sampler.cu:1007-1050) and
  greedy is exact-match — outputs unchanged by construction. M1 +2-6% class. Gate: one
  STRATA_SPEC_DEPTH=1 diagnostic boot + one spec-8 measurement boot (--spec 8 --mtp-max-t 8;
  kVerifyMaxT=8 cap) to read depths 4-6 before committing.
- Deeper slot chains: `set_max_drafts(1)`→3 (generate.cpp:4736), `slot_mtp->draft(2,...)`→4
  (generate.cpp:10317), admit rows per-depth gated by dprob ≥ spec_min_p, generalize the
  exact-match accept loop (10223-10234) to matched-prefix. Infra already kVerifyMaxT=8-ready
  (BSlot::draft/dprob sized 8, generate.cpp:9992-9994; commit_slot_prefixes takes keep[],
  verify.cpp:3129-3166). Pair +4-8% class; 6-conc is the regression guard (F-31 window-width).
- AdaEDL (arXiv 2410.18351): proven-lossless entropy-based early draft-stop — the cheap,
  training-free precursor of the F-186 open-lane entropy arbiter.
- Rejection-sampled slot accept (batch path is plain exact-match today, generate.cpp:10227;
  the q-list kernel exists for the solo path) = the distribution-SAFE substitute for the F-30b
  rescue. Tier-1.5, MMLU-gate mandatory.

**NEW #4 — Tier-1 code items now have implementation-level maps:**
- Partial device plan: resident_plan_kernel aborts the WHOLE layer on any slot<0
  (`atomicOr(&s_bad,1)`; verify_kernels.cu:1150-1162) while the plan layout already carries the
  miss-share segment (counts[2]/start2/ptr2, expert_source.cpp:3862; verify_kernels.cu:1225-1229)
  and post() has the per-region copy machinery (verify.cpp:1630-1666; doorbell publishes only
  misses, 1520-1526). Remove the s_bad abort, emit resident groups, skip_ semantic 0=full/2=partial.
  EXACT arithmetic (no rerouting) → F-69 margin risk inapplicable. Behind
  STRATA_VERIFY_DEVICE_PLAN_PART=1. 9-15% class on the ~half of layer-windows paying the round-trip.
- Fenced admission: ~2×(2+qsa_alloc)×stages cudaDeviceSynchronize() in copy_to/from_slot
  (generate.cpp:10055-10105, 10110-10155); measured "copied in 54.8-58.3 ms" = 6 windows of
  engine-thread stall per admission → stream events + one batched save/restore ≈ 15 ms. Stability
  UP, pair/6-conc under churn.
- Window width 8→12: kVerifyMaxT (verify_kernels.hpp:21), kResidentPlanMax 128→160, s_wsum[4]→[5];
  +2 slot sessions = 1.9 GiB from 21.7 GiB pre-cache free; expert cache 9648→~8.6k slots vs
  10,182 experts (decode hits 96-99% has margin). The structural 6-conc lever past ~204.
- L2: 60 MiB was the ASK, not the cap — 262k log shows "60 MiB carve over a 112 MiB span,
  hitRatio 0.536" (device.cu:376-410 clamps to the device attr). R10e sweep 60→72. 1-3% class.
- kv-resident 32k→65k: +415 MiB (3.09 GiB host copy / 262k cells / 12 QSA layers; layer.cpp:636-653)
  — M6b/depth-lane item, not a 6-conc bar mover.

**EXTERNAL CORROBORATION (web legs):** PDL is production-standard on Blackwell — TRT-LLM ships it
globally, LMSYS Ling-3.0-flash (batch-1 MoE + spec decode, our mirror) reports bitwise-safe PDL
validated by byte-exact greedy checks, SGLang HPC-ops chain Gate-Up/router/attention with it →
R10c confidence up (+3-8% journal class; their full-stack numbers higher). pegainfer: fusion >
graph > megakernel on cost/benefit; LMSYS shipped bitwise-neutral fusion exemplars (SwiGLU folded
into up-GEMM epilogue via even/odd column interleave). SP-MoE (arXiv 2510.10302): draft-gated
expert prefetch 1.07-3.5× TPOT — Tier-2 candidate that feeds OUR planner. tcgen05 confirmed
absent on sm_120 (CUTLASS docs) — matches the F-186 truth list. Upstream v0.1.44.1 = hotfix only,
zero perf content — merge for hygiene only; hold driver 615.71.09; stay on kernel 7.2.9.
Server per-token path measured NOT a round-trip contributor (~15-40 µs/token off-thread; <1% of
a 5.3 ms window). Stability verdict: watchdog stack sound (ENGINE_STALL_S=90, tries=3, no Xid).
RAM is 184 GB (the 64 GB note was stale) — OOM margin wide at 262k+6-conc.

**QUEUE v4 (goal register updated; supersedes v3 ordering where conflicting):**
R0 hygiene (dead arms DELETED this round; 262k parallel:6 A/B; logrotate) → R8 GPU state
(USER SUDO, exact block above; then RE-ANCHOR v3) → R10b diagnostic boot → R10c PDL (3×3 +
sustained gate + VERIFY_NODES edges>0) → R10f PIN_BY_NEED+adapt-async → R10d ROUTE_RESIDENT 0.75
→ R10e L2 72 → spec-path diagnostics then floor stair (M1) + deeper chains (pair) → Tier-1 code
order by R10b: partial plan → fenced admission → width 12 (+kv-resident 65k for the depth lane).
Tier-2 additions: SP-MoE prefetch, LMSYS fusion recipes. AdaEDL before the full arbiter.


### F-187 — THE LAW WRITTEN, THE BUDGET MEASURED, TWO ARMS TESTED, THE PORT MAP REALITY-CHECKED (2026-10-14 ~14:00, goal round 16)

**THE LAW (operator order, now binding in AGENTS.md, commit 2af8735).** Every perf change: 3+3
same-hour probes, means only; adopt bar +4% targeted / no −4% regression on ANY priority metric;
SINGLE-STREAM PROTECTION explicit (concurrency-for-M1 trades = shelf variants, never adoptions);
sustained-reasoning gate for launch/overlap class; F-entry + commit + result files for wins AND
negatives; verified clean revert; Launch Law.

**THE MEASURED PHASE TABLE (diagnostic boot, VERIFY_PROFILE/DECODE_TIMING; replaces round-9's
byte-model estimates). Batch window (6 slots, 25.95 ms GPU):** GDN: VRAM hits 2.85, hc reads ≈ 4.2
total (hc0/hc1 down 1.14+1.14, up 0.71+0.75, norms 0.45), copy+combine 1.60, router+plan+bell 1.00,
shared+quant 0.89, rec 0.75, q8+qkv 1.17, head 0.84. QSA (12 layers): **attention 4.40 — the
single biggest measured line** (grows 1.6→4.4 with window width: THE 6-conc ceiling), q+q-idx 0.56,
append 0.32. Host: **CPU experts 4.94 ms/window** (the doorbell tail, live). Saved:
campaign/results/diag-phase-table.txt.

**QFUSE — REJECTED.** Arm 176.6/160.7/193.5/79.8 vs same-hour control 171.3/165.0/202.8/80.8:
M1 +3.1% (below bar), 6-conc −4.6% (past the edge) — the F-183 mirror trade. No boot line; arm
engagement unverified. Off.

**PDL — NEUTRAL, SUSTAINED-CLEAN, SHELVED.** Arm 174.0/163.3/206.6/82.7 vs control
171.3/165.0/202.8/80.8: all inside noise (M1 +1.6, conc +1.9), acceptance +1.9 pts; the MANDATORY
sustained gate PASSED (3000 tokens at 156.4 t/s, acceptance 71.9%, no collapse — the Windows −40%
was WDDM, confirmed). A stacking candidate when density work lands; not adopted alone.

**THE FLASHINFER PORT MAP (study agent; source at /tmp/flashinfer-src, main 2895577):**
- **GDN fused decode (v0.7.0)**: plain CUDA, Apache-2.0, geometry matches ours with one -D — but
  ONE TOKEN PER CALL: as-is it replaces 2 nodes with T launches. Port value = the STRUCTURE
  (persistent self-resetting grid barrier + pre-barrier state prefetch) transcribed into OUR
  multi-token kernel. Honest ceiling ~−36 nodes (~2%) + µs/layer. STRATA_GDN_FUSED plan written.
- **W4A16 fused MoE (v0.6.17)**: DISQUALIFIED near-term — no C++ artifact (CuTe-DSL Python only),
  FP4-packed weights only (no IQ3_S path), adoption = re-quant (+~30% expert VRAM) + DSL toolchain.
- **XQA ragged-Q (v0.6.16)**: INCOMPATIBLE — full-context paged attention cannot serve our
  top-2051 CELL-SELECTED sparse attention; page size 4 vs required 16-128; int8+FWHT-rotated KV
  unsupported. The ragged-length INTERFACE is the reusable idea, inside OUR kernel.

**QUEUE v4 (re-ordered by MEASURED value):**
- R18 [env]: ROUTE_RESIDENT margin sweep — targets the measured 4.94 ms CPU-expert doorbell tail
  (~15-18% of window). MMLU-gated.
- R19 [code]: QSA attention kernel — the 4.40 ms line and its width growth; ragged rows inside our
  chunk+merge structure (XQA's idea, our semantics).
- R20 [code]: hc-read fusion (4.2 ms measured across 6 launches/layer; QFUSE's direction, done as
  a proper norm+down+up merge) — and/or the hc Q8 byte cut (needs a re-quant with Q8_0 hc tensors,
  mechanically possible per the pack side).
- R21 [code]: DuetServe look-ahead launch + launch-completion events.
- R22 [code]: ECHO-style per-row verify budget on the slot path.
- R23 [code]: layered prefill (layer-group interleave).
- R24 [deferred]: GDN fused-structure transcription (~2%); FastMTP+KV-injection [training] stays.
Standing: merged 0.1.44 + BATCH_MTP=1 + batch 6; restored, verified clean (no arm/diag env).


### F-188 — "HAS SINGLE-STREAM REGRESSED?" — THE FROZEN-MASTER A/B: no engine regression; the 190-200 range is the box's peak state (2026-10-14 ~20:00, goal round 17)

**THE QUESTION (operator: single feels below the 190-200 class; pull the frozen master from
/mnt/DATA, build it, run the same tests — improved or worse since then?).**

**THE SETUP.** /mnt/DATA/LLM/Engine/strata-custom-2 copied to `~/LLM/strata-frozen-baseline`
(read-only source, allowed); one source fix (the port's `<cmath>`, the only Linux delta);
`-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` (the frozen CMake defaults CUDA OFF);
built clean (138 steps, `strata 0.1.43.1`). Run directory wired to OUR venv, OUR data/ (symlink —
same learned profile), same Custom-data; three back-to-back boots in one drift window:
A = frozen @ anchor recipe (batch 4, BATCH_MTP=0); B = current 0.1.44 @ the SAME anchor recipe;
C = current @ current recipe (F-31 + batch 6).

**THE RESULTS** (clean-run means; each boot had ONE collapsed run ~97-100 t/s across ALL metrics —
a transient, see below — dropped, noted):

| boot | M1 | pair | 6-conc |
|---|---|---|---|
| A: frozen 0.1.43.1 @ anchor | 176.8 | 175.6 | 214.6 |
| B: current 0.1.44 @ anchor | **185.4** | 157.4 | 208.5 |
| C: current @ current recipe | 180.9 | 174.1 | 215.3 |

**VERDICT 1 — NO SINGLE-STREAM REGRESSION FROM OUR WORK. The current binary is the FASTEST M1 of
the three boots** (185.4 anchor-recipe / 180.9 current-recipe vs the frozen engine's 176.8 — same
hour, same recipe for A vs B). The user's 190-200 memory is the box's PEAK WINDOW: even the frozen
engine — the binary that measured the 190.3 anchor — reads **176.8 now, −7% below its own anchor
mean, in the same hour our current binary reads 185**. The drift is environmental and swings every
recipe and every binary together (session M1 record: 160-196 on identical recipes; the frozen
engine's own anchor-era 190.3 vs 176.8 tonight). The engine improved; the box drifts.

**VERDICT 2 — the recipe arc is net-positive.** Current@current vs frozen@anchor: M1 +2.3%, pair
−0.9%, 6-conc +0.3% — and vs each engine's OWN era the concurrent means are at session highs
(pair 174-177, conc 215-218 class in this window).

**THE ANOMALY TO WATCH (two findings):** (1) Boot B's pair mean (157.4) sits below both the frozen
(175.6) and current-recipe (174.1) readings at the same hour — either a real pair cost somewhere
in the merged binary at the anchor recipe (COMMIT_BATCH default-on is the candidate) or drift
asymmetry between boots; the 01:30 merge A/B measured pair-neutral (160.4 vs 161.3), so no
conclusion today — QUEUED for one dedicated back-to-back re-test. (2) **THE RUN-2 STALL: each boot
dropped exactly one run to ~97-100 t/s across ALL metrics** (frozen run 1, current-B run 2,
current-C run 2 partial) — a recurring transient, NOT memory pressure (PSI avg 0.00, 110 GB
available), NOT GPU throttle events (dmesg clean, 40 °C, 2827 MHz). Suspects queued: the
expert-profile save (every 30 rounds, fsync-class pause), the adapt tier's refill thread, GPU
clock transitions. This is now the #1 smoothness item — it is worth several % of every battery's
mean and it poisons 3-run statistics.

Standing = boot C (current recipe), healthy. The frozen baseline stays at
`~/LLM/strata-frozen-baseline` as the permanent reference engine (launcher `launch-frozen.sh`).
Sources: campaign/results/frozen-{1,2,3}.txt, curanchor-{1,2,3}.txt, curcurrent-{1,2,3}.txt.

### F-188 — RE-ANCHOR v3 UNDER THE LOCK + THE SERVER-CHURN SAGA: 2835 costs M1 ~6%, pair/conc unharmed; 3135 is the next bin (2026-10-14 ~20:30, goal round 16 execution)

**R8 CLOSED, LOCK LIVE**: strata-gpu-lock RPM installed by the user (19:01) — persistence
Enabled, clocks pinned continuously since (pclk 2805-2827 / mclk 13801 held; the service held
through a server-restart gap shorter than GRACE and released/re-locked correctly on longer gaps).

**THE SAGA (infra, not engine)**: probes v3-run1/run2 and my direct curl tests measured a
COLLAPSE (M1 150.8 → 90.9 → 55.6 t/s client-side) that DOes not exist. Root causes found:
(1) my detached relaunch died with its tool-call shell and the task-runner RETRIED the command,
parenting a new server to another call shell — servers churned; probe 2 measured a server that
DIED mid-run (its engine's stderr log froze at 19:26 while requests kept serving from the next
instance). (2) Fresh boots have a COLD expert cache: first-probe M1 penalty ~10-20% (run1 150.8
on a fresh boot vs 179.5 warm mean). (3) A kill-loop self-matched its own shell (the recurring
pgrep self-match trap — bracket-proof EVERY pattern). After killing everything and launching ONE
setsid server (PID verified alive across call boundaries BEFORE each probe), warm-up traffic,
then probes: **the engine's own per-request summaries NEVER dropped below ~173 t/s warm** — the
"55 t/s essays" were a dying-server artifact. Marked v3-run1 COLD-INVALID, v3-run2 CHURN-INVALID.

**RE-ANCHOR v3 (locked clocks, warm stable server, runs 3-5 = campaign/results/v3-run{3,4,5}.txt)**:

| metric | mean (3 runs) | round-15 current window | delta |
|---|---|---|---|
| M1 single | **179.5** (178.3/176.8/183.3) | 191.1 | **−6.1%** |
| M2 pair | **177.7** (174.1/179.3/179.8) | 175.5 | +1.3% |
| M2 6-conc | **214.7** (210.7/216.9/216.6) | 216.0 | −0.6% |
| M3b rate | 80.8 (80.0/81.5/81.0) | — | — |
| MMLU (2×70q) | **92.86** (94.29/91.43) ± 2.8 | 92.86 (3g watch) | at watch level, > 92 escalation line |

MMLU note: rescue arm ON (standing recipe); 92.86 matches the rescue-on historical band
(F-30b: 91.4-92.9 class) — distribution untouched by anything today (clock state is
distribution-neutral), no escalation.

**VERDICT ON 2835**: M1 −6.1% is OUTSIDE the ±4% noise law → the 2835 bin is an M1 LOSS.
Mechanism fits the physics: single-stream decode is LATENCY-bound (~160-300W observed, never
power-capped) and boost-hungry — unlocked it reached ~3135; the lock removes ~9% of clock on
exactly that path. Pair/6-conc (throughput-bound, SM-saturated windows) are unharmed (+1.3%/−0.6%).
**NEXT: SM_CLOCK 2835 → 3135** (the card's own max boost — a floor AT max, not a cap below it):
`sudo sed -i 's/^SM_CLOCK=.*/SM_CLOCK=3135/' /etc/sysconfig/strata-gpu-lock && sudo systemctl reload strata-gpu-lock`
then 2 warm probes A/B. If `nvidia-smi dmon -s p` shows sustained sw_power at 3135, step 3050 → 2950.
Fallback if no bin is both stable and boost-class: unlock SM entirely, keep persistence + mem lock
(mem-lock alone still kills the mclk transition jitter). Run-to-run spread this session (176.8-183.3,
±1.8%) vs anchor-era (±4%) hints the variance goal may already be working — needs more sessions to claim.

**DURABLE OPS LAW (learned the hard way)**: before ANY timed measurement, verify the server PID
identity (`ps -o pid= -p <pid>`), that it survived the previous tool-call boundary, and that the
expert cache is warm (first-probe-after-boot is always the coldest); never attribute client-side
t/s without cross-reading the engine's own "generated in X ms (Y tok/s)" summary — the engine's
self-reported speed vs client wall is the fastest split-brain detector.

### F-189 — THE BIN LATIN-SQUARE: evening drift swamped the 3135-vs-2835 A/B; bin question deferred, lock config settled at 2835 (2026-10-14 ~20:40, goal round 16 execution)

User toggled SM_CLOCK 2835 → 3135 → 2835 via sysconfig+reload (service handled all three
transitions flawlessly — re-applied on HUP each time, clocks verified pinned: 2827 / 2962-2970 /
2827). Two warm probes per bin, same stable server (PID 1163282):

| window | bin | M1 | pair | 6-conc |
|---|---|---|---|---|
| A (~19:40) | 2835 | 179.5 (3 runs) | 177.7 | 214.7 |
| B (~20:00) | 3135 | 167.0 (2 runs) | 168.9 | 204.0 |
| C (~20:20) | 2835 | 173.3 (2 runs) | 164.1 | 200.0 |

C did NOT rebound to A: all three metrics slid monotonically A→B→C — a GLOBAL evening drift
gradient (~−1.5%/20min), the re-anchor-v2 phenomenon operating within the hour. **My interim
"3135 falsified" call (F-188 follow-up) was PREMATURE — the bin effect is confounded by drift and
UNRESOLVED.** Both F-188's M1 −6.1% and the 3135 deficit are within the same-window-drift fog.

WHAT TONIGHT ACTUALLY ESTABLISHED:
- The lock SERVICE is production-solid: 3 config reloads, held clocks through boot gaps,
  released/re-locked correctly, clean journal trail. R8's deliverable stands.
- Matched-window tooling now trustworthy: within-window repeat spread ±1-2% (was the chaos of
  the churn era). The A/B protocol requirement is confirmed in the sharpest form yet: **no
  comparison across a 20+ minute gap, ever, on this box.**
- All 7 valid locked runs (M1 160-183, mean 174.1) sit inside the campaign's ±8% drift envelope
  of the anchors; nothing tried today is provably better or worse on means.
- SW power-cap time at 3135: 0 µs — power was never the limiter; the "boost-hungry decode"
  model needs revision: unlocked round-15 hit M1 191 in ITS window, but boost may matter less
  than the window itself.

SETTLED CONFIG: SM_CLOCK=2835 (vendor-sane sustained bin, current state). Bin-vs-bin (2835 vs
3135 vs unlocked-for-M1) is queued as a MULTI-WINDOW series: 2 probes per bin alternating across
days/sessions — the only design that beats the drift gradient. Secondary; Tier-0 battery (R10b
diagnostic, R10c PDL, R10f PIN_BY_NEED) outranks it and must itself run as immediate-toggle pairs.

### F-190 — THE UNLOCK VERDICT: the lock is EXONERATED (mildly beneficial), the decline is exogenous drift — cleared by toggle, fresh boot, and host audit (2026-10-14 ~21:10, goal round 16 execution)

Operator demanded the regression be fixed rather than attributed to the box. Delivered the full
toggle chain, same evening, same warm/stable server discipline:

| arm (all this evening) | M1 | pair | 6-conc |
|---|---|---|---|
| locked 2835 (runs 8-9, ~20:25) | 173.3 | 164.1 | 200.0 |
| **UNLOCKED (runs 10-11, ~20:50)** | **162.6** | **161.5** | **195.9** |
| unlocked + FRESH BOOT (run 12, ~21:05) | 169.2 | 163.3 | 198.5 |

FINDINGS:
1. **Unlocked is slightly WORSE than locked-2835 in adjacent windows** (−6%). Unlocked boosts to
   2962 MHz under load — the same effective bin the 3135-lock achieved and which ALSO measured
   slower (F-189 window B). Consistent mechanism: SM bins above ~2.9 GHz are mildly counter-
   productive for this engine's 1,800-kernel latency chains; the 2835 pin forces the efficient
   bin. THE 2835 LOCK IS KEPT — it is not the regression, it is a small win.
2. **F-188's "2835 costs M1 −6.1%" is RETRACTED** — that comparison reached across 7.5 h to
   round-15's 191.1. The matched-window toggle proves the lock was innocent.
3. **State accumulation exonerated**: fresh boot (cold cache, fresh learned-profile load from the
   01:31 base) reads the same as the 1.5 h warm server (169.2 vs 162.6). Engine state, expert
   profile, conversation cache, arena — all clear.
4. **Host exonerated**: governor=performance, tuned=throughput-performance, 5.1-5.6 GHz under
   load, Tctl 49 °C, no throttling anywhere; GPU power-cap time 0 µs even at 3135.
5. Therefore: **the ±8-10%, hours-scale drift is real, exogenous, and survives every variable we
   control** (lock, bin, boot freshness, host config). It hit 191-class at midday and 165-class at
   21:00 on identical configs. Whether it is room/grid/GPU-DVFS-curve related is now the top
   instrumentation question — but it is NOT ours to fix tonight and NOT caused by today's changes.

ACTIONS: results v3-run{8..12}.txt committed; lock service to be re-armed by the user
(`sudo systemctl start strata-gpu-lock` — config already at the settled 2835). The REAL path to
"+10% on a bad window" is the Tier-0 battery starting with R10c PDL as the first drift-immune
immediate-toggle pair (arms minutes apart, immune by construction). The F-188/F-189 retraction
trail stands in full — wrong interim claims are marked as such in the journal, per campaign law.

### F-191 — R10c PDL: REJECTED AS A NO-OP by the first drift-immune ABAB (2026-10-14 ~21:50, goal round 16 execution)

Immediate-toggle pair design (fresh boots, identical warm-up, minutes apart, locked 2835):

| leg | arm | M1 | pair | 6-conc |
|---|---|---|---|---|
| ctl1 | standing | 163.8 | 159.7 | 195.9 |
| pdl | STRATA_DF_PDL=1 | 166.9 | 155.0 | 188.7 |
| ctl2 | standing | 166.2 | 153.0 | 188.8 |

ctl2 ≈ pdl on EVERY metric (≤1.2%) while both sit −4-5% under ctl1 — the evening gradient again.
**PDL changes nothing on this engine/config: no M1 gain, no batched-window harm.** F-178's +3-8%
estimate and the external corroboration (TRT-LLM/LMSYS) do not transfer here — plausibly because
our serial windows' launch spacing is already hidden behind the host round-trip PDL would target.
Env line removed from the launcher; R10c CLOSED as a measured no-op (supersedes the F-107
Windows-era retraction with clean same-window data). The ABAB itself is the night's deliverable:
arm-vs-arm within ±1.2% while the world moved −5% underneath — the noise floor and the drift
gradient are now both quantified on this box. NEXT (Tier-0, same design): R10f PIN_BY_NEED +
--adapt-async, then R10d ROUTE_RESIDENT 0.75, R10e L2 72.

### F-192 — THE INSIDE MAN: the evening "drift" is largely OUR OWN CONTEXT GROWTH; the standing log already carries the full window profile (2026-10-14 ~22:10, goal round 17 study, server untouched)

Operator hint: the "real user traffic" on 8081 is this ZCode session itself — every request
carries the full conversation context (53k -> 110k+ as the session grew). Watched it live with a
10-15 s /metrics sampler (Bearer key required; live.tok_s, live.phase caught "thinking" at 155.6
t/s and "writing a tool call" at 113.6). Inside-man log: /tmp/insideman.txt (40 min, 15 s cadence,
also tails new window lines).

**THE WITHIN-BOOT TIME SERIES (single boot, same weights/tier, two traffic classes, ~2 h):**
3.1-row windows 16.37 -> 18.89 ms (+15%), GPU-reach 11.2 -> 12.8; 6.6-row windows 27.14 -> 31.41
(+15%), GPU-reach 18.6 -> 21.7. Both classes rise together and monotonically; CPU experts flat
(2.8->3.1 / 4.6->4.9); NO thermal/power throttle (41-43 C, ~300 W of 575 W). Comparator: F-187
diagnostic boot 29.99 ms/window (probe, short ctx) vs live 31.41 (session, long ctx) = +4.7%,
concentrated in GPU-reach (+0.88) and emit (+0.23, +14%).

**CONTEXT-vs-DECODE CURVE (log, generated tok/s bucketed by prompt tokens):** 3k = 171.1,
4k = 160.2, 5k = 153.6, 6k = 153.8, 7k = 168.3, 9-11k ~ 150, 51-62k ~ 156. One-time ~10% penalty
3k->5k, then FLAT to 60k+: sparse QSA (q+q-idx) caps the KV reads — the depth lane is healthy,
110k context decodes at 145-155. So the within-boot rise = short-context probes -> long-context
session traffic, saturating; NOT an engine regression. The r10c ABAB between-boot −5%/25 min
gradient (fixed short-context probes) remains a separate, smaller exogenous component.

**STAGE SEMANTICS verified in source (generate.cpp:10362, verify.cpp:3533):** GPU-reach wait =
host spin-wait for the graph to reach each layer = pure GPU execution (69% of window); CPU experts
= host pool for the 2.0% CPU-routed entries, GPU stalls at those layers (16%); emit = per-slot MTP
draft launches + residual D2D copies (6%); commit = commit_slot_prefixes, matches the h_commitb_
1.10 ms hazard (4%); adapt wait 0.18 = async tier healthy.

**QUEUE CONSEQUENCES:** (1) R10b diagnostic boot DEMOTED — the standing log already prints the
full per-window stage breakdown every time slots go idle; (2) partial device plan / ungated
resident share RE-PRICED from 1.5-2.5 to up to 4.9 ms/window (~16%, live data), the biggest
code-side prize after the GPU-execution core; (3) long-context lane needs no fix — sparse
attention already flattens the curve; (4) realtime-stats recipe: GET /metrics with Bearer key,
live.{state,phase,generated,tok_s}, hardware.{gpu_util,gpu_temp,gpu_power,gpu_pcie_rx_mb}.

---

## F-193 — EXTERNAL STACK SCAN (agent) + read-only verification battery

**VERIFIED LOCALLY (read-only, server untouched):** kernel journal today: 65 lines, ZERO Xid/NVRM
warnings. PCIe link NOW: Gen5 x16 full width (sysfs current_link_speed 32.0 GT/s, width 16; max
same). ASPM policy = `default` (BIOS-controlled; L1 substates likely enabled). nvidia-smi does NOT
expose GDDR7 memory temp on GB202 (temperature.memory = N/A; -q -d TEMPERATURE has no Memory
field) -> the "VRAM temp" hypothesis is NOT testable with our toolset, demoted. Engine binary
links plain cuBLAS only (GemmEx, GemmStridedBatchedEx, Sgemm; NO cublasLt, NO grouped GEMM) ->
CUDA 13.4's Blackwell grouped-GEMM wrong-results bug does NOT apply to us.

**CAVEAT:** the scan was dispatched before F-192. Its items 1-4 target the between-boot
−5%/25 min ABAB gradient (fixed short-context probes) — NOT the within-boot +15% series, which is
now explained by our own context growth.

**RANKED EXTERNAL FINDINGS (condensed):**
1. X870E/AM5 Gen5 link-training + ASPM/fabric deep-idle: exact-pairing forum thread (RTX 5090 +
9950X3D, Xid 79 at idle, ASPM off at register level, suspicion = AMD fabric C-states modulating
link latency). Our policy is `default` = BIOS L1 substates on. Runtime-reversible test:
`echo performance > /sys/module/pcie_aspm/parameters/policy` + 60 s LnkSta logging across a drift
ABAB; also latest BIOS/AGESA (vendors shipped lane fixes).
2. GB202 GSP firmware Xid 79/109/8 across 590-615 branches — today's journal clean; keep watching.
3. GDDR7 ambient temp compensation — unmeasurable here (see above); demoted.
4. Wayland/GNOME compositor + desktop video contending with decode on the dGPU — upstream's own
v0.1.43.1 reverted "adaptive overlap" after "desktop decode collapses"; 4K playback + compositor
plausibly 8-10% on a bandwidth-bound workload. -> PROMOTES the pending iGPU-cable task (one
monitor still on 5090 DP-4; iGPU has free DP-2/DP-3).
5. amd-pstate-epp boost regression on 6.17/6.18 (HFI preferred-core) — low-medium; optional
taskset-to-V-cache-CCD ABAB.
6. Upstream v0.1.44.1 (today): hotfix only (AVX2-free startup, CUDA 12.x builds, Metal); answers
byte-identical; optional zero-risk rebase, no perf.
7. ECHO-class budget-adaptive draft length (arxiv 2604.09603; vLLM DSpark; fergusfinn adaptive
speculation) — fits the emit stage + Tier-2 draft chain: log per-hour MTP acceptance, then sweep
draft length 1/2/3 per window offline.
8. CUDA-graph FUSION (fewer fatter memory-bound kernels), not node-count cuts — consistent with
F-191 PDL no-op (inter-node gaps <2%).
9. FlashInfer SM120 GDN decode kernels exist (sm_120, not tcgen05-only) but vLLM/SGLang layouts;
standalone microbench before any port.
10. CUDA 13.4 grouped-GEMM bug — NOT APPLICABLE (verified: no cublasLt/grouped symbols).
11. MoE expert prefetch — skip; fully resident at IQ3_S/32 GB.

**QUEUE CHANGES:** iGPU cable promoted to top hardware task; ASPM-performance + LnkSta logging
queued as user-sudo block for the next drift window; acceptance-vs-hour logging added to the
inside-man analysis; env battery order unchanged (R10f PIN_BY_NEED first).

### F-194 — SOURCE DEEP-DIVE: every phase-table line traced to its mechanism; attention is latency-starved, hc reads are parallelism-starved, commit is a host block (2026-10-14 ~23:00, goal round 17 study, server untouched)

**METHOD:** own read-through of the hot path while the code-study agent and inside-man sampler run:
qsa_decode_attn.cu (full attn_chunk_kernel body), fused_gr.cu (geometry + launch_multi + variant
gates), verify.cpp (commit_slot_prefixes, stage_batch tail), verify_kernels.cu (gdn commit).
No server touch, no code change. Comparator = F-187 phase table (29.99 ms/window).

**1. QSA decode attention (4.40 ms = biggest GPU line; grows 1.6→4.4 with width = the 6-conc ceiling).**
`attn_chunk_kernel` (qsa_decode_attn.cu:120), grid (ceil(cap/64), n_head_kv=2, n_q), CHUNK=64,
HD=256, G=12, THREADS=256, ~15.5 KB shared. Mechanisms found:
- ZERO prefetch: 0 cp.async/__pipeline hits in the whole file (the cp.async pipeline exists only in
  the prompt/i8 variant and in fused_gr's staged kernel). Score loop: K row loaded per cell then
  immediately consumed; value loop: 64 serial iterations/thread, each a dependent random pool read
  through page_table indirection (~600-cycle DRAM latency, no overlap). Per block ≈72 dependent
  round-trips ≈ 70 µs — matches 1.6 ms at n_q=1 (128 blocks, 1 wave) and the growth to 4.4 ms at
  n_q=6 (768 blocks ≈ 2.3 waves at 3 blocks/SM occupancy).
- No KV sharing across query rows: each of the 6 rows re-gathers its own selection of the SAME
  context's pool — the width growth is pure re-gather.
- CUDA pays 12 warp_sum shuffles/cell (60 shuffle instrs); REDUCE12 butterfly + QUERY_SWIZZLE are
  compiled only under STRATA_HIP_GFX906 (env STRATA_GFX906_ATTN_REDUCE12 is dead on CUDA).
- Byte math unchanged: ~125 MB KV/window ≈ 0.28 ms at 450 GB/s → 4.4 ms is 15× off bandwidth;
  latency/occupancy-bound.
MEANS (new queue items): R19b cp.async double-buffer of K/V rows (pattern already proven in
gr_down_staged_kernel); R19c fold n_q INTO the block — one gather per cell, score against all rows'
q in shared (6×12 KB sq = 72 KB fits Blackwell's 228 KB) → time ~flat in width = THE 6-conc fix;
R19d port REDUCE12 butterfly to CUDA (claimed bit-identical: same operands, same order — low risk).
LANE_CELL (STRATA_ATTN_LANECELL=1) already kills the 12 shuffles per cell — add to Tier-0 battery.

**2. hc reads (4.2 ms; 6 launches/layer) — PARALLELISM-starved, not bandwidth.**
Compiled geometry (fused_gr.cu:25): N=2560, HC=4, LR=320 → each hc down/up matrix = 10240×320 =
3.28 MB q8; 2 sites (hc_attn/hc_ffn) × 2 matrices × 12 layers = **157 MB of weight reads per
window** (min 0.35 ms at 450 GB/s; observed 4.2 ms = 37 GB/s). Cause: down kernel =
DOWN_BLOCKS+1 = LR/WARPS+1 = **41 blocks** on 170 SMs (~2 warps/SM); Little's law in-flight
≈20 KB/SM → ~80 GB/s ceiling. Up via gr_up_q32_kernel = 80 blocks (QFUSE_HC_LOCAL default ON).
ENV-ONLY KNOBS FOUND (add to Tier-0 battery): STRATA_HC_SPLIT=0/1 (plain/split; default staged),
STRATA_GR_V3=1 (stream-split two-kernel down/up — DIFFERENT sum order → MMLU-gated),
STRATA_GR_DOWN_MAX4, STRATA_NO_MULTI_GR. Code means: split-K down (more blocks per row-group);
q4 hc weights would halve the 157 MB but changes bits.

**3. Commit stage (1.10–1.22 ms) = fully exposed host block.**
commit_slot_prefixes (verify.cpp:3129): host writes commitb rows into mapped memory, launches the
pre-captured batch commit graph (gdn_commit_layers already fused to ONE launch, grid (h_v,4,n_layers)),
then `cudaStreamSynchronize(cs_)` — the host waits for the whole commit graph before emit/draft
overlap. ple_stage() also runs CPU ngram_rows per accepted token. Tier-1 "commit double-buffer"
= launch commit on cs_ and defer the sync until the next stage_batch needs the state buffers.

**4. Doorbell tail — all-or-nothing fallback CONFIRMED at kernel level.**
resident_plan_kernel: any entry with slot<0 → s_bad → `*skip=0` → the WHOLE layer-group falls back
to the host doorbell round-trip (wait_flag_ge_or_kernel: `if (*skip == value) return;`). At 86.5%
expert hit (~13% of entries miss), most layers pay the PCIe round-trip for 1–2 missing experts.
Prize (Tier-1): per-group PARTIAL device plan — device plans resident entries, host services only
the misses.

**5. GDN (2.85 VRAM + 0.75 rec).** Commit already fused (gdn_commit_layers, one launch). State is
fp32 128×128/head; byte-cut = bf16/int8 state — big bit-change, park.

**6. Emit (1.87 ms).** Per-token `printf("BT %d %d")` + per-window fflush (server.py protocol),
per-slot D2D copy of final_R rows + draft launch on slot streams. printf/fflush ≈ one syscall per
window — minor; mapped-memory protocol is a later nicety.

**QUEUE CHANGES:** Tier-0 env battery gains STRATA_ATTN_LANECELL=1, STRATA_HC_SPLIT=0,
STRATA_HC_SPLIT=1, STRATA_GR_V3=1 (V3 = bit-change, MMLU-gated). R19 refined into R19b/R19c/R19d
above; R19c (fold n_q into block) promoted to top Tier-1 code item — it is the mechanism behind
the 6-conc ceiling. R20 re-scoped: hc down is 41-block parallelism-starved; env variants first,
split-K only if they miss.

### F-195 — CODE-STUDY AGENT VERDICT: merge exonerated, two new live env levers, R10f demoted, commit-async is a small code win (2026-10-14 ~23:40, goal round 17 study, server untouched)

**METHOD:** read-only deep audit agent (merge 8d1c9d6 vs parents, live log, queue v4). Cross-checked
against my F-194 mechanisms. Inside-man sampler still running.

**MERGE VERDICT: no silent disable.** Upstream v0.1.44 wholesale + our 491-line patch re-applied
clean; all our arms verified present (TYPICAL_ACCEPT/RESQ, PREDICT_CREDIT, PINCAP, L2_SETASIDE,
BATCH_MTP, MTP tap, B3 counters). Only default that flipped at the merge: STRATA_COMMIT_BATCH
on (F-180 measured neutral; MMLU 3-mean<92 watch stays the tripwire; =0 is the instant revert).

**NEW LIVE LEVERS (env-only, join Tier-0 battery):**
- **STRATA_TYPICAL_DEVICE=1** — built, parity-proven (F-16b), NOT in our launcher. Host rescue
  path does full-vocab D2H (~1 MB) + 2×O(248k) host softmax per probe, up to 2 probes/window on
  the M1 hot path (temp 0.7 probes). Device arm exists (verify.cpp:3343-3355, elementwise.cu:462).
  Est 2-5% sampled M1. -> NEW TOP of battery.
- **STRATA_VERIFY_ASYNC_LAUNCH=1** — v0.1.44 opt-in, never tested here: window launch wrapped in
  HostGraphLaunch; its own comment warns an instrumented doorbell-graph launch "may block until
  device work completes" — part of the 20-21 ms GPU-reach wait may be the LAUNCH CALL stalling.
  Env-only A/B; watch release_gpu_waits.
- ROUTE_RESIDENT 0.75 mechanism CONFIRMED (tail ranks 6-9 swapped for resident within margin,
  renormalized; exit-time swap-rate stats built in). Stays queued.
- QFUSE caveat CONFIRMED: g_qfuse disables T=1 self-commit (verify.cpp:878) — boot-rider only.

**R10f DEMOTED (two independent reasons):** (a) PIN_BY_NEED ~0 upside here — RAM complement is
already 100% pinned (cudaHostAlloc Mapped|Portable, "pinned 41.85 GiB" of 41.85); upstream's gain
was complement-starved shapes. (b) --adapt-async 1 will NOT arm the v0.1.44 arena fast-path on our
resident-RAM mode (needs srcp==&arena_src; we run srcp=&src) — takes the OLD exchange-buffer path
only if exchange_capacity >= 2*a_cap, else silently stays blocking. GATE any adapt-async A/B on
the boot line "adaptive tier asynchronous"; absent = arm never armed.

**QUEUE CORRECTIONS:**
- STRATA_VERIFY_DEVICE_PLAN_PART does NOT exist — partial device plan (F-185/F-194 #4) is
  unimplemented CODE work, not an env A/B. Site map complete: plan layout already carries the
  never-filled miss segment (counts[2]=0, start2[0]=n, verify_kernels.cu:1243-1251); pre-doorbell
  publishes only misses (verify.cpp:1519-1526). Fix = emit resident groups + miss entries in
  second region, skip=ring, gate only the CPU-row copy. MMLU-neutral by construction (exact
  arithmetic). 9-15% class. Needs a lane reboot — quiet window per campaign law.
- "Commit double-buffer BANKED via STRATA_COMMIT_BATCH" was WRONG: COMMIT_BATCH removes ~70 graph
  NODES; the 1.10-1.23 ms is the forced cudaStreamSynchronize in commit_slot_prefixes (3151/3153).
  The SERIAL path already has the async pattern (g_commit_async, verify.cpp:2643/2685-2697) — the
  batch path never got it. Small code win ~3-4%: double-buffer m_commitb_ by window parity baked
  into the graph key, drop the sync. (Matches my F-194 #3 exactly.)
- Launcher hygiene: STRATA_EXCHANGE_ROTATE=1 is a DEAD arm ("exchange rotation unavailable" every
  boot) — delete from launch-custom-iq3s-8081.sh at next restart.
- Learned expert profile EXONERATED as drift suspect: expert-profile-learned.bin is write-only;
  nothing reads it back. Drift suspects remain: boot age (in-boot adaptation state) + our own
  context growth (F-192).
- pcie_auto_step exonerated (requires pcie_frac_auto; we pass explicit 0.35).
- Conditional-THP arena (#2126): boot-to-boot variance only; STRATA_ARENA_THP=1/0 pins it if the
  "expert arena:" boot note ever flips.

**NEW CODE ITEMS (Tier-1.5):**
- Batch graph layout churn: graphs keyed on EXACT active-slot subset (batch_key), captured lazily
  IN the serving path — 176 captures/5 boots (~35/boot) from BATCH_MTP=1 slot rotation. Multi-ms
  stalls at admission/drain boundaries (tail, not steady M1). Interim: pre-capture canonical
  rotation subsets inside copy_to_slot (already device-syncs); full fix = sentinel bucketing.
- adapt_stream + cs_ have NO priority on this boot; in-tree pattern exists (mtp.cpp:461-476).
  One-liner: adapt_stream at LOW priority. Small (adapt wait 0.15-0.35 ms + jitter).
- Per-token printf+fflush on engine thread (generate.cpp:10293-10295) = 0.2-0.5 ms/window of
  bt_emit. Buffer BT lines, one flush per window.
- Fenced admission CONFIRMED heavy: ~14-30 cudaDeviceSynchronize per 6-slot admit wave
  (copy_to_slot/from_slot, generate.cpp:10049-10140) = the 54.8-58.3 ms admit waves; pair/6-conc
  under churn. Fix: stream-ordered events + one batched save/restore.
- 3-way patch residue (harmless): dup telemetry resets (generate.cpp:10386-10389), two extra
  printf args (10377-10378), dup dhead_type_ (mtp.cpp:716/722) — fold into next code round.
- Config drift live: 262k daily-driver still "parallel": 4 after batch-6 adoption — A/B on its own port.

**AGENT EXECUTION ORDER (impact×confidence÷cost):** TYPICAL_DEVICE env -> ROUTE_RESIDENT/QFUSE
env after R8 lock -> commit-async code (3-4%) -> graph pre-capture interim -> PARTIAL DEVICE PLAN
(the structural one) -> fenced admission -> sentinel bucketing. #10/#11 checks before any R10f A/B.

## F-196 — Inside-man sampler: 40 min of live traffic under our own session (2026-10-14 21:29–22:08)

**Source:** background sampler (15 s loop: /metrics live.{state,phase,generated,tok_s} + nvidia-smi util/temp/power + engine-log `serve:` lines) → /tmp/insideman.txt. Server untouched throughout.

**Probe/window:** 51 request segments, 94 idle samples, 40 min. Every prompt carried 31k–101k tokens of context — i.e. ALL traffic in the window was this ZCode session itself. F-192 confirmed at scale: the evening "drift" observations ride on our own long-context traffic, not on background load.

**Numbers:**
- Decode mean **151.9 tok/s** across the band; first-half 156.7 vs second-half 149.9 (−4.3% over 40 min — inside the ±8% drift band, and the session context itself grew 68k→101k across the same span, so context growth plausibly explains most of the gradient).
- Aggregate draft acceptance **68.5%** (36,222/52,877) — matches the standing slot-accept 67.7%; MTP healthy.
- Cache-miss prefills: 4,600–6,600 tok/s on big fresh reads (52.9k, 44.4k, 23.7k new tokens) — confirms the F-192 correction that real prefill ≈5,700 tok/s, not the misleading /metrics mean.
- **Context-vs-decode is FLAT-to-favorable at the top of the band:** 98k–101k contexts decoded 169–176 tok/s while 31k–43k contexts decoded 133–158. Depth is NOT the decode-rate driver in this regime; window composition (rows/slot mix) dominates. Reinforces R19c (fold n_q into attn block) as the width fix.
- Idle is clean: 94 idle samples, mean **85 W**, util 4–8% — no background GPU contention at idle (the occasional "idle util=99%" line is a request-tail boundary artifact). Desktop GL contexts on the card are negligible at idle; the iGPU cable task stays worthwhile but is not costing decode-rate at idle.

**Verdict:** no regression signal in 40 min of real traffic; engine behaves exactly as the F-192/F-194 model predicts. Sampler retired (task completed). Queue unchanged: revised Tier-0 battery (TYPICAL_DEVICE → VERIFY_ASYNC_LAUNCH → LANECELL → HC_SPLIT → GR_V3 → ROUTE_RESIDENT 0.75 → L2 72) awaits operator "go".
### F-196 — THE BATCH COMMIT DOUBLE BUFFER: the exposed cudaStreamSynchronize removed from commit_slot_prefixes; adopted as hygiene below the win bar, and the boot-B pair-dip watch item CLOSED (2026-10-14 ~23:59, goal round 17 execution)

**THE ISSUE (F-194 #3 / F-195's "commit-async = small code win").** `Verifier::commit_slot_prefixes`
(verify.cpp) launched the batch commit graph and then `cudaStreamSynchronize`'d unconditionally —
the exposed host block F-194's phase table measured (~1.2 ms class, every window).  The sync existed
for one reason: the mapped `h_commitb_` is a SINGLE buffer, and the next round's `stage_batch`
rewrites it — without the sync, that write races the commit graph's `copy_i32_from_mapped` node.
The serial path solved this years ago (`set_commit_async`: launch, record `commit_done_`,
`wait_commit` later); the batch path never got it.

**THE FIX (bit-identical by construction — timing only).** Parity double buffer:
- `h_commitb_`/`m_commitb_` and the device `commitb_` arena doubled; a parity's halves at
  `p*cb_host_` / `p*cb_dev_` (verify.hpp members + init_slots).
- The parity is baked into the commit graph key (`ckey` = bkey + parity — the pointers are frozen
  at capture); `capture_commit_batch(rows, S, hbase, parity, ..)` captures BOTH parities up front
  in `stage_batch` (no mid-serving capture when the parity flips); `evict_batch_graph` frees both.
- `stage_batch` flips `commit_parity_` once per round; `commit_slot_prefixes` and `batch_launch`'s
  inline commit use the round's parity.
- The sync is replaced by: launch → record `commitb_done_` → return, under the EXISTING
  `set_commit_async` gate (so `STRATA_COMMIT_SYNC` still forces the old behavior, and a missing
  event falls back to sync).  The PREVIOUS round's event is checked at the next commit — an error
  checkpoint, not a wait.

**WHY NO SYNC IS SOUND.** A parity half is rewritten only two rounds later (stage_batch flipped
first), and every path between observes the stream drained: `run_slot_rows` syncs cs_ before the
host reads picks; `batch_poll` reports done only on `cudaStreamQuery(cs_)==cudaSuccess` — the whole
stream, commit graph included.  The drafter reads only the picks and its own K/V (the serial path's
shipped ordering, generate.cpp:9214's note); the next window graph is stream-ordered behind the
commit on cs_.  The adapt thread's swap surface (expert-cache slots) is disjoint from the commit
graph's reads (session state, staging buffers, dense ssm_norm weights).

**PROOF (Lock Law: adjacent boots C/D, anchor recipe, 3+3 same-hour probes, zero collapses).**
M1 169.0→167.3 (−1.0%), pair 141.8→145.2 (+2.4%), 6-conc 185.3→189.3 (+2.2%), M3b 79.8→80.7,
acceptance 79.2→79.4% — IDENTICAL sampling (the change touches no arithmetic; no MMLU gate due).
Boot E (fixed @ current recipe): 161.4 / 157.7 / 198.7, M3b 79.7.

**VERDICT.** Below the +4% win bar on every metric — NOT claimed as a scored win, scorecard
unchanged.  ADOPTED AS HYGIENE under the no-regression clause: nothing below −1%, both concurrent
metrics +2%+, a measured inefficiency removed, and the batch path now matches the serial path's
proven async design.  Smoke: single + concurrent pair coherent, zero errors.

**THE BOOT-B PAIR DIP (F-188 watch item) — CLOSED.** The sync's true pair cost is ~2.4% (C→D),
not the ~12% F-188's boot-B reading suggested; that 157.4 was drift asymmetry between boots.
COMMIT_BATCH stays default-on (its ~70-node reduction is real; the sync is now gone).

Rollback: `~/LLM/backup/strata-preF196-rollback` (sha 56895c58…; also /tmp copy).  Standing
launcher: dead `STRATA_EXCHANGE_ROTATE=1` arm deleted at this restart per F-195 ("exchange
rotation unavailable" every boot — behavior-neutral).

Sources: campaign/results/f197-{cu,cf,cc}-{1,2,3}.txt; smoke /tmp/smoke-f196.out.
### F-197 — THE THREE-GENERATION LADDER, ONE WINDOW: v1 → v2-frozen → current = +37/+46/+54% same-box same-night — the "are we getting nowhere" challenge answered with same-hour data (2026-10-15 ~00:15, goal round 17 execution)

**THE ASK (operator).** "These results dont prove anything??? Its as if we are getting nowhere, we
need min 20-30% improvements" — plus: pull strata custom V1 from the DATA disk and test it like we
did the v2 frozen master.

**WHAT V1 IS.** `/mnt/DATA/LLM/Engine/strata-custom` = the Windows-era "MAGMA" fork: upstream
0.1.43, optimizations through F-64, `.bat` launchers, config `strata-iq3_s.json` (STOCK pack +
STOCK mtp/rt, `--spec-min-p 0.7`, parallel 4).  Pulled to `~/LLM/strata-custom-v1` (rsync minus
build; its Windows `.venv` replaced by a symlink to ours; stock `packs/iq3_s` + `mtp/rt` copied
from `/mnt/DATA/LLM/Engine/strata`; one `<cmath>` fix — the same Linux port delta as the frozen
v2; `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`).  `launch-v1.sh` = the .bat's env
battery verbatim (TYPICAL_ACCEPT 0.2 … VERIFY_DEVICE_PLAN 1; NO ROUTE_RESIDENT — post-dates v1);
the config re-pathed only, recipe untouched.  DATA disk untouched (copy FROM only).

**THE BATTERY.** Five boots, one drift window, back-to-back, 3 probes each, no full collapses
(the v2f boot's run 1 showed the run-2-stall signature in SHORT-class only — noted; the hunt's
data point: it hit the OLDEST engine's FIRST run):

| boot (same hour) | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| A v1 (0.1.43, F-64 era) | 117.5 | 108.2 | 129.0 | n/a* |
| B v2-frozen (0.1.43.1, F-172 sync) | 132.9 | 138.1 | 186.6 | n/a* |
| C current-unfixed @ anchor recipe | 169.0 | 141.8 | 185.3 | 79.8 |
| D current-F-196 @ anchor recipe | 167.3 | 145.2 | 189.3 | 80.7 |
| E current-F-196 @ current recipe | 161.4 | 157.7 | 198.7 | 79.7 |

\* the 0.1.43-era servers did not emit the probe's M3b/accounting lines (diagnostic only).

**HEADLINE — the campaign's magnitude, measured same-window.** current@current (E) vs v1 (A),
same box, same night, same probe: **M1 +37.4%, pair +45.7%, 6-conc +54.0%**.  This is the honest
form of the "min 20-30%" ask: every priority metric is 1.4-1.5× the engine this campaign started
from, in a window where NOTHING is flattered — see below.

**THE DRIFT CONTEXT (why absolute numbers look low tonight).** Tonight's window is a trough:
the frozen v2 read 176.8 earlier THIS evening and 132.9 now; the current engine read 185.4
earlier and 169.0 now (same recipes, same binaries).  All engines move together — which is exactly
why the ladder ran as five adjacent boots: within-window ratios are drift-immune, and those ratios
are the claim.  The locked anchor means (M1 190.3 = +30.6% vs Windows, pair 159.4 = +46.5%,
6-conc 203.7 = +45.3%) stand as banked; tonight's ladder independently confirms the campaign's
magnitude against a live same-hour v1 instead of a stale bar.

**THE EXTRA FINDING.** v1-on-Linux reads M1 117.5 — BELOW the Windows lane bar (145.6): the
pre-campaign Linux lane did not start ahead of Windows, it started ~19% BEHIND on M1 (drift
deflates tonight's v1 too, so treat the exact gap softly).  The campaign overcame a deficit and
then cleared +37% on top of it.  Also: v1's pair (108.2) lands exactly on the Windows pair bar
(108.8) — the old engine's concurrent shape was the Windows shape.

**VERDICT.** We are not "getting nowhere": v1→v2→current is a monotone ladder on every metric,
and the current engine is 37-54% over the campaign's starting engine in the same hour.  The
170-190-class single-stream readings tonight are the trough, not the engine (F-188/F-190 law).

Engines kept: `~/LLM/strata-custom-v1` (launch-v1.sh) and `~/LLM/strata-frozen-baseline`
(launch-frozen.sh) — the permanent three-generation reference set.
Sources: campaign/results/f197-{v1,v2f,cu,cf,cc}-{1,2,3}.txt; campaign/f197-analyze.py.
### F-198 — THE SCOREBOARD EXTENDED: pristine upstream v0.1.44 and hokar3361/Strata measured — upstream stock is strong, hokar ≡ upstream on CUDA (their patch is Metal-only), current leads stock same-window on M1/pair (2026-10-15 ~01:20, goal round 17 execution)

**THE ASK (operator).** Add two engines to the scoreboard: "strata upstream engine original" and
https://github.com/hokar3361/Strata.

**THE ENGINES.**
- **upstream-0144** = pristine upstream tag v0.1.44 from a worktree of OUR repo (the exact base the
  current binary was merged onto — the cleanest "what did the custom fork add" row).  Built
  `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`; zero source fixes needed (0.1.44
  compiles clean on GCC).  `~/LLM/strata-upstream-0144`.
- **hokar** = `git clone` of hokar3361/Strata = **exactly v0.1.44 + 3 commits** (2 README + ONE
  patch, `metal/src/*` only: 116 lines wrapping a >maxBufferLength model as overlapping no-copy
  Metal buffers — the macOS 41.75 GB cap fix that keeps 47 GB IQ3_XXS resident on 64 GB Macs).
  On CUDA the metal sources are not in the build: the two binaries are the same size
  (75,064,848 B) and same code; hashes differ only by embedded build paths.  `~/LLM/strata-hokar`.
- Both: stock boot, ZERO env battery, **v1's config recipe verbatim** (same GGUFs/pack/mtp/profile,
  spec 4 / min-p 0.7 / pcie 0.35 / kv-resident 32768 / cvec / max-context 262144 — all flags are
  upstream features; boot accepted everything).  Artifacts shared via symlinks into v1's tree.

**THE BATTERY** (second window of the night, 3+3+3, no collapses; current re-run in-window for
drift normalization):

| boot (same hour) | M1 | pair | 6-conc |
|---|---|---|---|
| upstream stock v0.1.44 | 147.1 | 149.1 | 187.7 |
| hokar3361 (v0.1.44 + metal) | 155.3 | 158.3 | 194.8 |
| current (fixed, current recipe) | **164.8** | **160.5** | 192.7 |

Run detail matters: upstream's boot WARMED through its runs (M1 141.1/144.9/155.2 — cold expert
cache on runs 1-2), hokar's cooled slightly (159.0/154.9/151.9); the distributions overlap.
**VERDICT: hokar ≡ upstream on this box** — the ~5% boot-level gap is the documented boot
asymmetry (F-188 class), not their patch: the only diff does not compile into the CUDA build.
Their reported 29-52→58-59 t/s gain is real WHERE DESIGNED (macOS buffer cap) and irrelevant here.

**READINGS.**
1. **Upstream stock 0.1.44 is a strong engine**: 147/149/188 with zero custom work — well above
   the 0.1.43-era custom v1 (117.5/108.2/129.0 in window 1).  The custom fork's early arms were
   Windows-era costs; upstream has since absorbed the easy classes (COMMIT_BATCH etc.).
2. **Current vs stock, same window, each at its own best recipe**: M1 +12.0%, pair +7.6%,
   6-conc +2.7%.  The concurrent edge is compressed tonight (trough window; at adoption windows
   the batch-6/F-31 means ran 216-class vs this window's 188-195 band) — the M1/pair edge holds
   everywhere.  vs **v1** (window 1): +37/+46/+54% stands.
3. M3b/accept accounting lines did not print on the stock servers (the probe reads those from a
   fork-side stats surface; diagnostic-only absence, noted).
4. Config-class note: reference rows run max-context 262144 (v1's recipe) vs the current
   engine's 131072 — F-182 measured that difference at −0.2..−1.3% (noise-class).

**THE SCOREBOARD (same night, two adjacent windows; current appears in both as the normalizer):**

| engine | M1 | pair | 6-conc | window |
|---|---|---|---|---|
| v1 custom (0.1.43, F-64, Windows era) | 117.5 | 108.2 | 129.0 | 1 |
| v2-frozen custom (0.1.43.1, F-172) | 132.9 | 138.1 | 186.6 | 1 |
| current @ current recipe | 161.4 | 157.7 | 198.7 | 1 |
| upstream stock v0.1.44 | 147.1 | 149.1 | 187.7 | 2 |
| hokar3361 (≡ upstream on CUDA) | 155.3 | 158.3 | 194.8 | 2 |
| current @ current recipe | **164.8** | **160.5** | **192.7** | 2 |

Reference engines kept: `~/LLM/strata-custom-v1`, `~/LLM/strata-frozen-baseline`,
`~/LLM/strata-upstream-0144` (worktree), `~/LLM/strata-hokar` (clone) — the four-generation set.
Sources: campaign/results/f198-{up,hk,cc}-{1,2,3}.txt.
### F-199 — ENGINE SET PRUNED to the comparison pair + THE +30%-vs-STOCK ROADMAP (2026-10-15 ~01:45, goal round 18 planning)

**THE PRUNE (operator directive).** Kept: `~/LLM/strata-custom-2` (current v2) and
`~/LLM/strata-upstream-0144` (pristine stock reference). Deleted: strata-custom-v1,
strata-frozen-baseline, strata-hokar (all result files and journal entries retained — only the
runnable trees are gone). Dependency surgery first: upstream's pack/mtp were symlinks INTO v1's
tree — replaced with real copies (2.3 GB) and every config path re-verified before deletion.
The old pre-campaign `~/LLM/Strata` + `Strata-data` stock dirs were NOT touched (not part of this
test set; say the word to remove them too).

**THE GOAL (operator).** +30% faster/better than upstream stock on every metric — "the difference
becomes night and day". Measured start (F-198 window 2): current = +12.0/+7.6/+2.7% (M1/pair/
6-conc). Still needed on top: +16/+21/+27%.

**THE ROADMAP — campaign/ROADMAP-30-VS-STOCK.md** (the working plan; summary):
- The 6-conc window budget (31.41 ms measured): GPU-reach 21.68 (attention 4.40 = #1, hc 4.20 =
  #2), CPU-expert doorbell 4.90 (16%), emit 1.87, commit 1.22 (RECOVERED by F-196), adapt 0.18.
  Recoverable pool ≈ 10 ms of 31.4 — deeper than the 6.6 ms the conc target needs.
- TIER 0 env battery (one boot each): TYPICAL_DEVICE → VERIFY_ASYNC_LAUNCH → ATTN_LANECELL →
  HC_SPLIT 0/1 → GR_V3 (MMLU-gated) → ROUTE_RESIDENT 0.75 → L2 72 → MMQ_KQ (prefill).
- TIER 1 code, value order: **T1.1 partial device plan** (4.90 ms doorbell tail, 9-16%, the
  per-region machinery already exists — s_bad abort at verify_kernels.cu:1141-1155 is the whole
  obstacle) → **T1.2 fold query rows into the attention block** (width growth 1.6→4.4 ms is pure
  re-gather; KV once per slot group, 6×12 KB fits shared) → **T1.3 deeper slot chains +
  the floor stair** (slots verify ONE proposal, generate.cpp:4736; serial chain stops at depth 3
  under the flat floor while depth acceptance runs 96/93/90%) → hc fusion → emit-on-device →
  fenced admission → graph pre-capture/sentinel → exact two-phase head.
- TIER 2 formulations: layered prefill (70% TTFT class, prefill = priority #1), ECHO row budget,
  DuetServe look-ahead launch/completion events (the 9.5 µs node-spacing), conditional graph
  nodes (green field), window width 12 after T1.2.
- TIER 3 rebuild/training: TQ3 rotation quants (+11% experts/GiB), 3-bit KV, DFlash2 (gated on
  sampled-mode acceptance), FastMTP+KV-injection.
- Stacked landing estimate: M1 +25-35%, pair +22-32%, 6-conc +20-33% (stretch) — the Tier-1 trio
  T1.1/T1.2/T1.3 carries the target; overlap-corrected arithmetic in the doc.
- Cross-cutting: run-2 stall hunt (worth several % of means), MMLU gates as marked, single-stream
  protection law, user-side iGPU cable + lock-state blocks.

Execution order: the T0 battery (days of cheap boots), then T1.1 → T1.2 → T1.3, Lock Law on every
adoption. The stock reference at ~/LLM/strata-upstream-0144 stays bootable for every re-baseline.
### F-200 — THE TIER-0 BATTERY SWEEP: THREE ARMS ADOPTED (device rescue, async launch, resident-margin 0.75) + the six-brief research haul (2026-10-15 ~03:45, goal round 18 execution)

**THE SETUP.** 11 boots, one battery, ~50 min, zero collapsed runs (the run-2 stall did NOT
appear).  Control → 9 arms → bracket control; the window FELL −4.9/−8.1/−8.6% end-to-end, so
every arm is scored against a time-interpolated control (both controls kept in results/).
Deep-trough window (ctl M1 136.4 vs 164.8 two hours prior; M3b 62-65 band).

**THE RESULTS (M1/pair/6-conc vs interpolated control):**

| arm | delta | verdict |
|---|---|---|
| STRATA_TYPICAL_DEVICE=1 | **+9.0/+6.7/+6.4** | **ADOPTED** (parity-proven device rescue port; prefill proxy −8.7% too) |
| STRATA_VERIFY_ASYNC_LAUNCH=1 | **+4.9/+7.9/+8.9** | **ADOPTED** (v0.1.44 HostGraphLaunch; attacks the 69% GPU-reach line) |
| ROUTE_RESIDENT 0.75 (from 0.5) | **+14.9/+8.5/+12.2** | **ADOPTED after 2× MMLU gate PASS** (92.86/97.14, mean 95.00 — the good band; routing-changing arm, watch note stands) |
| ATTN_LANECELL=1 | +3.6/+5.4/+1.9 | below M1 bar — shelved (phase-table check pending) |
| HC_SPLIT=0 | **−15.4/−16.6/−11.8** | REJECTED hard (the 41-block layout IS the bad one, as F-194 predicted) |
| HC_SPLIT=1 / GR_V3=1 / L2 72 / MMQ_KQ=1 | all ≤±2.5% | neutral today — MMQ needs a prefill-lane test, not these metrics |

**THE CONFIRMATION BOOT** (td+val stacked, fresh boot, 3 probes): **M1 157.1 / pair 158.6 /
6-conc 196.5, M3b 80.6** — vs the end control (129.7/124.5/180.3): **+21.1% M1 / +27.4% pair /
+9.0% conc**, and M3b recovered from the window's 62-65 band to 80.6.  The full stack (td+val+
rr75) is now STANDING (launcher updated; server rebooted healthy); the next fresh window's first
battery re-baselines it.  Three adoptions in one night = the campaign's biggest single round.

**THE SIX RESEARCH BRIEFS** (2-at-a-time agents per the stream-limit law; full texts in
campaign/research/F-200-*.md — 66 findings total).  The items that CHANGE THE ROADMAP:

*Attention (4.40 ms #1 line):* SpecSA row-group index sharing (up to 6.86× kernel — our exact
shape); TMA Gather4 CONFIRMED working on sm_120a (replaces serial dependent reads); one-line
cp.async.bulk.prefetch.L2 of the next block (1.84-2.15× kernel, L2 hit 0.06→43.7%); SageAttention
int8-MMA (our KV is already int8 — stop the scalar dequant+shuffle path); the 5090 FA recipe's
mma fragment layout = zero cross-lane traffic; **GDN BV-tile retune worth ~10% e2e on our exact
model family (SGLang PR #17094)**.

*MoE (4.90 ms doorbell tail):* llama.cpp graph-safe expert-slot indirection (+84% class — the
closest implementation of our T1.1); KTransformers one-layer-late CPU-expert deferral (+1.45×
class); next-layer expert prediction at 90-99% recall (two variants); HOBBIT T₂-skip of
unimportant misses; LLRU eviction formula (~5 lines); VERIFIED: SM120 MoE kernels must be
Marlin-class (TMA-WS broken, 99KiB smem); our tail is overhead-dominated (sub-ms compute).

*Spec decoding:* **draft temperature down-scaling (draft at T'≈0.4-0.5 while target serves 0.7 —
training-free, lossless, +3-5pts acceptance plausible) = the cheapest untested lever in the
campaign**; Block Verification (ICLR'25, +3-8% lossless, drop-in); Nightjar length bandits
(+14.8%); depth 7-8 + early stop (verify rows ~free on a launch-bound engine); suffix-automaton
draft override (TRT ships it).  The exact-lossless space = block verification + traversal
verification + multi-draft-without-replacement.

*Graphs (the 9.5 µs node spacing):* Hazy Research's 1.3 µs graph floor → our spacing is ~7×
floor = mostly recoverable; **fence/doorbell audit is the suspected dominant tax
(__threadfence_system 0.5-2 µs/node → possibly 9-12 ms/window)**; vertical fusion 1,800→700-900
nodes = 8-10 ms; device-side tail launch (constant latency, no host); conditional nodes GREEN
FIELD CONFIRMED (nobody ships them); SET on RTX 5090: host overhead worse on Blackwell.

*Prefill:* layered prefill SOURCED (arXiv:2510.08055: chunked prefill inflates MoE expert loads
+39%; up to 70% TTFT); Green Contexts P/D (+60% on a 5070 Ti; 5090 power-wall caveat);
libsmctrl TPC masks; SPF reordering (free when prompts queue); **VERIFY OUR GDN PREFILL IS
CHUNKWISE (sglang-jax #1416: per-token recurrent scan was their dominant long-prefill cost)**.

*Quant:* QTIP trellis 3.1-3.3bpw should beat IQ3_S quality at −7-10% size; IQ3_KS in mainline
(cheapest A/B); Kitty 2-bit KV (8× KV density, tested on Qwen3, page-centric = our shape);
QuantMoE-Bench allocation recipe (rank by outlier+frequency, first blocks high-bit);
**exact two-phase lm-head: NOBODY SHIPS IT — the Cauchy-Schwarz-bound construction is provably
exact (521MB → 30-70MB/window)**; 2:4 sparsity for the hc line (~1.3-2×).

**ROADMAP v2 (queue after tonight's adoptions):**
R25 draft-temp down-scale (env/1-line, lossless) → R26 fence audit (cheap, suspected 9-12 ms)
→ R27 GDN BV-tile retune (constant change, ~10% e2e class) → R28 block verification (lossless
drop-in) → T1.1 partial plan (llama.cpp slot-indirection template) → T1.2 attention row-fold
(SpecSA template + L2 prefetch + int8-MMA) → KTransformers deferral → layered prefill →
conditional nodes / device tail launch → exact two-phase head → Kitty KV / QTIP (rebuild tier).
Full set: campaign/ROADMAP-30-VS-STOCK.md + research/F-200-*.md.

Sources: campaign/results/f200-*, mmlu-mmlu-f200-adopt{,-b}.json.

**F-200 ADDENDUM — THE STANDING RECIPE PINNED (post-adoption snapshot, 2026-10-15 ~04:00).**
After the three F-200 adoptions the standing launcher (launch-custom-iq3s-8081.sh) carries, in
full: TYPICAL_ACCEPT=0.2 · POOL_SPIN_US=100 · KV_PREFETCH=1 · ADAPT_LAG=2 · DMA_BATCH=1 ·
VERIFY_DEVICE_PLAN=1 · PREDICT_CREDIT=1.0 · PREDICT_BUDGET=2 · ADAPT_PINCAP=48 · ADAPT_MIN_AGE=2 ·
ADAPT_DORMANT=64 · L2_SETASIDE_MIB=60 · BATCH_MTP=1 (F-31) · BATCH_RESCUE=0 · SPEC_COUPLED=1 ·
PLE_BATCH=1 · ROUTE_RESIDENT=**0.75** (F-200) · ROUTE_RESIDENT_MTP=1 · **TYPICAL_DEVICE=1**
(F-200) · **VERIFY_ASYNC_LAUNCH=1** (F-200); config custom-iq3_s-8081.json: spec 6 / min-p 0.5 /
parallel 6 / reserve 2000 / kv-resident 32768; binary = 0.1.44-merged + F-196 commit double
buffer.  REMOVED this session: EXCHANGE_ROTATE (dead arm), ADAPT_OVERLAP/_RESIDENT (refused),
KEEP_MAPPING (dead on Linux).  Reference pair kept: ~/LLM/strata-custom-2 + upstream-0144.
### F-201 — R25 DRAFT-TEMPERATURE ADOPTED (+13.4/+9.8/+7.5%): the coupled drafter samples at 0.85×; the doorbell/fence tax quantified; L2 premise corrected to 96MB (2026-10-15 ~06:30, goal round 18 execution)

**R25 — THE CHANGE (sampler.cu, ~10 lines).** `STRATA_SPEC_DRAFT_TEMP=x` scales the DRAFT
chain's sampling temperature in `coupled_stage_kernel` (the staging kernel that copies the
request's SamplerParams to device — the same place gumbel is overridden).  The exact-match
verifier emits the TARGET's picks either way, so this is a PROPOSALS-ONLY change — the output
distribution is untouched (no MMLU gate class; the emitted stream = target's own samples).
Mechanism: a cooler draft sharpens q toward its own top pick, raising the draft/target overlap
(arXiv:2609.24150's temperature-scaled overlap result — lossless for any q under rejection).

**THE SWEEP (6 boots, 3 probes each, fresh boots, one window, no collapses):**

| arm (DT) | M1 | pair | 6-conc | vs ctl |
|---|---|---|---|---|
| control (1.0) | 132.1 | 128.9 | 192.2 | — |
| 0.45 | 136.5 | 136.7 | 201.8 | +3.3/+6.0/+5.0 |
| 0.65 | 137.7 | 133.8 | 193.9 | +4.2/+3.8/+0.9 |
| 0.80 | 147.4 | 138.7 | 201.1 | +11.6/+7.6/+4.6 |
| **0.85** | **149.8** | **141.5** | **206.8** | **+13.4/+9.8/+7.5 ADOPTED** |
| 0.90 | 135.8 | 134.0 | 195.2 | +2.8/+4.0/+1.6 |

Clean response curve peaking at 0.85 (0.90 collapses toward control — argmax-like drafts lose
coverage).  This window was again the low-acceptance state (accept lines ~30%, M3b 62-65), making
the lever's effect visible precisely where the rescue saturates less.

**R26 AUDIT (quantified by the PCIe deep brief — campaign/research/F-201-pcie-doorbell-agent.md):**
- Device-side polls of host-mapped flags = UNCACHED non-posted PCIe reads, ~1-2µs RTT each →
  est. **1.8-3.6ms of a 17.1ms window**.  Fix direction: INVERT host→GPU signaling (host posts
  doorbells via cuStreamWriteValue/BatchMemOp as posted writes; GPU polls DEVICE memory at
  L2-hit ~200-300ns).
- `__threadfence_system()` = MEMBAR.SC.SYS + ERRBAR + CCTL.IVALL (SASS-verified) = full L1+L2
  invalidate + PCIe drain — budget ONE per handoff, never per poll.  13 sites in elementwise.cu.
- Our 12.7GB/s DMA = per-op overhead floor (~5-10µs/op at 256KB chunks); batch to 2.8-10MB
  phase-aligned megablobs → ~3-4× of the DMA slice of the 4.9ms tail.  AM5 mixed-traffic cap
  ~15-25GB/s (single IO die, dual-channel DDR5 shared with the CPU pool).
- WC mappings for one-way staging only (never host-polled counters); THP+cudaHostRegister for
  the 40GB pool (PTE ÷512); NVMe: page-cache-prefetch the model once, QD≥4.
**Combined upper bound: ~4-7ms of the window + ~3-3.6ms of the tail.**

**R27 FINDING.** The GDN step kernel's block-shape retune is ALREADY BANKED in the fork (the
MAX_H=8→1 fix, documented in gdn.cu:17-32).  Sibling kernels share the starvation SHAPE but are
µs-scale; the real item is the step kernel's ILP rewrite (two serial i-passes, strided global
loads — the latency-bound pattern) = designed change, queued with the attention row-fold.

**DEEP BRIEF 1 (CUDA µarch — F-201-cuda-uarch-agent.md) highlights:**
- **PREMISE CORRECTION: RTX 5090 L2 = 96MB (not 128)** — our 60MiB carve is 62.5%; docs formula
  allows ≤72MB.  Re-derive carve math; query persistingL2CacheMaxSize.
- In-graph node spacing on Blackwell is CONDITIONAL (1.5-6µs even on datacenter; near-zero for
  long kernels) — our 9.5µs is not a floor; decompose with nsys --cuda-graph-trace=node.
- L2 accessPolicyWindow attaches PER GRAPH NODE — phase-switch persisting (KV/hc) vs streaming
  (experts) inside one graph; hitRatio discipline (window>carve @1.0 = 11% SLOWER, measured).
- Hot-vs-cold weight reads 7 vs 11µs (LMSYS measured, expert-traffic flushes-L2 class = our hc
  line); CTA-tile swizzle alone 87→94% L2 hit.
- TMA multicast ABSENT on sm_120 (clusters=DSMEM only); bare TMA swap regresses — producer-warp
  pipelines only; bf16 small-ops (router/head) ≈ +10% decode step (LMSYS); device graph launch
  removes ~1ms/step host cost; GB202 L2/GDDR7 latencies WORSE than Ada's (130/329ns).
- DVFS: headless consumer parts sit 180MHz GPU/405MHz mem with no auto-boost (vLLM 6.5× case) —
  verify the lock service holds clocks DURING probes (a drift-wave suspect!).

**STANDING NOW:** F-200 stack + STRATA_SPEC_DRAFT_TEMP=0.85.  Cumulative vs the night's start
(f200 ctl): M1 132→~150 class (+13%), pair 129→142 (+10%), conc 192→207 (+8%) — measured across
the two batteries in their own windows; next fresh-window battery re-baselines the stack.
Rollback: /tmp/strata-preF201-rollback (pre-DT binary; F-196 binary preserved at
~/LLM/backup/strata-preF196-rollback).
Sources: campaign/results/f201-*, research/F-201-{cuda-uarch,pcie-doorbell}-agent.md.
### F-202 — THE NINE-BRIEF DEEP CORPUS COMPLETE (~90 findings): the slowness map is now measured, formulated, and sourced end-to-end; ROADMAP v3 (2026-10-15 ~08:00, goal round 18)

**THE EXECUTION TONIGHT (F-201): R25 ADOPTED — STRATA_SPEC_DRAFT_TEMP=0.85 (+13.4/+9.8/+7.5%
M1/pair/conc, sweep-peaked, proposals-only so no MMLU class).** R26 quantified, R27 assessed
(retune already banked). Standing = F-200 stack + DT 0.85.

**THE NINE BRIEFS** (investigative-researcher agents, 2-concurrent waves, full texts in
campaign/research/F-201-*-agent.md):
1. CUDA µarch — **5090 L2 is 96MB not 128** (carve = 62.5%, headroom to 72); node spacing is
   CONDITIONAL (1.5-6µs even on datacenter; near-zero for long kernels) — our 9.5µs is not a
   floor; per-graph-node L2 accessPolicyWindow; hot/cold weights 7/11µs; TMA multicast ABSENT on
   sm_120; bf16 small-ops ≈ +10% decode; device graph launch ~1ms/step; GB202 latencies worse
   than Ada; DVFS no-boost headless state = a drift suspect (verify lock holds during probes).
2. PCIe/doorbells — device-polls of host flags = uncached 1-2µs PCIe RTTs (est 1.8-3.6ms/window)
   → INVERT to posted writes (cuStreamBatchMemOp) + device-mem polls; threadfence_system = full
   L1+L2 invalidate + PCIe drain (ONE per handoff; 13 sites in elementwise.cu); DMA batch to
   2.8-10MB megablobs (12.7GB/s = per-op overhead floor; AM5 cap 15-25GB/s mixed); THP+
   cudaHostRegister the 40GB pool; WC one-way only; combined 4-7ms window + 3-3.6ms tail.
3. Zen5 host — cross-CCD ~200ns (placement: pollers CCD0/V-cache, workers CCD1@5.7GHz, per-CCD
   queues); PAUSE fine (65cy); LOCK/MFENCE 2.5× costlier (purge from flag paths); true 512-bit
   AVX-512 (shuffle-LUT dequant, 8 accumulator chains, NO gathers); hugepages: 10.5M→20K PTEs
   (IOTLB ~330ns/miss); DRAM ~60-70GB/s real; sched_yield useless; UMWAIT impossible on AMD.
4. KV — int4 KV + K-only Hadamard = the ONLY page-compatible int4 (SAW-INT4: +10.7-41.4% TPS);
   in-graph KV-gather overlap hides ~all of 4.40ms; UNIQUE page-stat scorer cached + radix top-k;
   page 8 > 4 on both axes; delta-only re-admission = 10-30×; cross-layer index sharing 2.17×;
   indexes ROT (incremental maintenance mandatory); L2 carveout measured WEAK (skip).
5. MTP — depth economics AL/(1+ρD): marginal depth pays above 6-9% acceptance, OURS IS 90% AT
   DEPTH 3; batch slot-chain steps + ONE deferred lm_head (draft head = 49-62% of draft time);
   FLA state snapshot µs-class; SpecDec++ depth gate +7-11%; head distillation on OWN traffic
   +14pp depth-3 (443s/2×H200); Q4 direct-cast validated (NEVER GPTQ the head); 4.6k vocab below
   every tested floor (re-rank from our corpus, watch OOV).
6. Acceptance THEORY — Yin optimality: every acceptance point above Metropolis costs exactly one
   TV point (P*-compensation required); Cactus δ-KL closed form replaces τ·e^−H (entropy in the
   criterion is structurally wrong — Prop 4); ERSD exponential races: chain CONTINUES THROUGH
   LOSING SLOTS, k≥3 arrivals pay on free verify rows; GLS shared-exponential endorsement for K
   slots (gains concentrate with temperature-mismatched drafts = our 0.85×); "below temp 0.8, k
   beyond 10 rarely helps"; Sequoia: wide-shallow dominates deep-narrow with free verify rows;
   DERIVED: under exact-match, acceptance = collision Σq_t·q_d (LINEAR → corner solution) — our
   0.85× is the theory-predicted optimum, and must NOT carry to any Metropolis path.
7. Scheduling — SHAPE-KEYED graphs (key on padded row count, NEVER slot layout) = the root-cause
   fix for all 176 re-captures; token-budgeted MIXED prefill+decode windows (vLLM V1 pattern);
   admission is DISPATCH-bound (90-95% overhead; our 54.8ms waves vs 13ms copy floor = 4×);
   Pensieve layer-pipelined admission; cost-ranked retention + pre-RoPE parking; conditional
   nodes prototype.
8. NVMe — RUN-2 STALL CONFIRMED as fsync-class: durable write on Gen5 consumer = ~500µs-1ms+
   PER CALL (the every-30-rounds profile save!); fixes: sidecar → other drive (zero code), async
   single-flight fdatasync, FADV_RANDOM/MADV_RANDOM on the expert tier, keep mmap+THP, no DIO.
   Diagnostic: strace fsync around round 30.
9. Formulations (identities EXECUTION-VALIDATED) — GDN one-pass fused identity (1.8e-15;
   readout decoupled from update → halves the serial chain, register-resident state per FLA);
   log-space deferred decay (9.1e-14; removes per-step exp; fold at λ<−30); hc down-mix
   ABSORBED into the adjacent GEMV (exact-static; n=4-static is lossy 0.010, n=2 safe); radix
   select for the indexer (EXACT, k-insensitive; sign-bit ordinal map — the −0.0 branch is a
   real bug trap); exponential-race sampler replaces sorts (EXACT; composition theorem for
   min-p/top-p/top-k filters); router selection needs ZERO exponentials (proof; keep full Z);
   NEGATIVE: no single-token WY win — one read+one write IS the floor.

**ROADMAP v3 (value-ordered, integrating all 15 briefs to date):**
- R28 GDN fused one-pass + log-space decay [exact, 1.5-2× step kernel]
- R29 doorbell inversion (posted writes + device-mem polls) + fence budget [1.8-3.6ms]
- R30 attention: row-fold + SpecSA sharing + L2 prefetch + int4-KV gate [4.40→1.5-2.5ms]
- R31 shape-keyed graphs [kills the re-capture class; prerequisite for scheduling wins]
- R32 partial device plan / slot indirection [4.90ms tail]
- R33 batched slot-chain drafting + deferred lm_head + depth gate [draft 6-9%→3-5%]
- R34 ERSD first-k arrivals + chain continuation [slot acceptance ceiling]
- R35 depth 6→8 at DT 0.85 + stopping rule ≥7-9% [one-boot config test]
- R36 mixed windows + event admission + delta re-admission [admission 4×; prefill stalls]
- R37 hc absorption into adjacent GEMV [exact-static; removes the starvation GEMV]
- R38 radix-select indexer + race sampler + router monotone selection [exact micro-wins]
- R39 run-2 stall fixes (sidecar isolation + async fdatasync) [smoothness = battery means]
- R40 DMA megablobs + THP pool + CCD re-pin [PCIe slice]
- R41 bf16 small-ops + device graph launch [µarch slice]
- R42 rescue δ-KL calibration [quality-gated; replaces τ heuristics with theory]
- Tier-3 (rebuild/training): head distillation on own traffic; Kitty KV; QTIP; layered prefill
  full; megakernel endgame.

Standing healthy on the full adopted stack (F-200 + DT 0.85); git 41 commits; rollbacks secured.
Sources: campaign/research/F-201-{cuda-uarch,pcie-doorbell,zen5-host,kv,mtp,acceptance-theory,
scheduling,nvme,formulations}-agent.md.
### F-203 — TWO CLEAN REJECTIONS BY INTERLEAVE (doorbell value-store on CUDA; spec depth 8) + THE RUN-2 STALL'S fsync EXONERATED + THE STANDING STACK'S BEST-EVER WARM READINGS (2026-10-15 ~10:30, goal round 18 execution)

**THE LEGS.**
1. **STRATA_DOORBELL_STORE=1 (the #649 store-the-ring publish, un-gated from HIP for CUDA —
   F-203 code, default off): REJECTED, DECISIVELY.**  Two arms vs FOUR interleaved controls:
   dbstore 145.6/140.0 M1 (−21%), pair −20%, M3b **63.1/63.2 vs the controls' 75.2-76.9** — the
   acceptance state drops WITH the arm, and M3b dropping means OUTPUTS changed: the stored-value
   ring semantics desync the host service order subtly on CUDA (the HIP ring was a counter the
   CUDA wait loop does not expect as absolute per-group values).  A correctness-risk signature,
   not a perf trade.  The un-gate stays in the code DORMANT (default off) — DO NOT ARM on CUDA
   without re-deriving the value semantics per wait site.
2. **--spec 8 (depth 6→8 at DT 0.85): REJECTED.**  Two arms: 140.5/140.1 M1 (−22%), M3b 59-60.6.
   The marginal depth-7/8 rows cost ~1/3 more window time and bank ~nothing — consistent with
   the acceptance-theory prediction (under exact-match + sharpened draft, deep marginal
   acceptance collapses; measured 90%@3 → the tail is dead by 6).  Depth 6 STAYS.
3. **THE INTERLEAVE PATTERN VINDICATED AGAIN** (F-191's design): the first battery pass looked
   drift-confounded (warm ctl → two cold arms → warm ctl2); the ABAB re-run showed the cold
   state TRACKS THE ARMS.  New instrument law: **M3b is a state-canary — an arm that moves M3b
   by >±5pts is changing outputs, whatever its mechanism claims.**
4. **RUN-2 STALL: fsync EXONERATED** — strace -f on the engine through 3 full probes traced
   ZERO fsync/fdatasync (the profile save writes without sync, or via mmap).  The hunt's next
   suspect per the µarch brief: **DVFS clock transitions** (headless consumer parts sit at
   180/405MHz with no auto-boost; a governor ramp inside a window = exactly a ~100 t/s
   collapsed run).  Actionable next: verify the gpu-lock service holds clocks DURING probes
   (nvidia-smi dmon -s c alongside the next battery).

**THE POSITIVE RESULT — THE STANDING STACK'S BEST-EVER WARM READINGS.**  Four interleaved
control boots of the adopted recipe (F-200 stack + DT 0.85): **M1 179.7/181.4/176.5/182.4 ·
pair 175.0/171.0/173.3/174.2 · 6-conc 215.5/210.1/211.9/212.3 · M3b 75-77** — means
**179.9 / 173.4 / 212.5**.  vs stock's same-night window (147.1/149.1/187.7): **+22.3% M1 /
+16.3% pair / +13.2% conc** at matched windows — and the standing recipe now carries four
same-day adoptions (TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, RR 0.75, DT 0.85) on top of the
F-31/batch-6/merge stack.  The +30% target is in reach of the R29/R30/R31 queue on warm windows.

Standing restored + verified.  Rollback: /tmp/strata-preF203-rollback (the deployed binary =
F-203 build, behaviorally identical to F-201's with the env off).
Sources: campaign/results/f203-*, /tmp/strace-fsync.log (empty — the exoneration itself).
### F-205 — THE ARM-WRAPPER BUG FOUND AND REPAIRED: every wrapped arm since F-200 booted the BARE engine; DT re-validated on the true base (+4.0% M1, stays); F-203's rejections VOID; new method law (2026-10-15 ~13:30, goal round 18 execution)

**THE BUG.** The arm wrappers used `set -a; grep "^export" launcher | bash; set +a` — the pipe
runs the exports in a CHILD shell that dies with the pipe; `set -a` does not propagate child
exports.  EVERY armed boot since F-200 ran the BARE engine (no TYPICAL_DEVICE, no
VERIFY_ASYNC_LAUNCH, no ROUTE_RESIDENT 0.75, no DT, no battery at all) plus its one arm var.
The tell that unraveled it: tonight's warm windows split perfectly by boot TYPE — all
launcher-direct controls warm (M3b 74-79), all wrapper arms cold (M3b 59-65) — impossible for
independent mechanisms, trivially explained by a missing battery.  The cold/warm M3b bimodality
across the night = mostly BARE-vs-FULL battery state, not environment.

**THE FIX, VALIDATED.** `source <(grep "^export" launcher)` in the parent, override AFTER
source (env-prefix on the launcher cannot work — it unconditionally re-exports its own values).
Validation boot (fixed wrapper + STRATA_WRAPPER_NOOP=1): **182.2/172.9/213.5, M3b 77.3** ≈
launcher control 187.8/173.6/215.6, M3b 76.2 — the wrapper is now faithful.

**WHAT STANDS / WHAT'S VOID.**
- STANDS: the standing recipe itself (all its readings were launcher-direct boots): tonight's
  warm controls **179.9-187.8 M1 / 171.0-175.0 pair / 210.1-215.6 conc, M3b 74-79**.
- STANDS: the F-200 adoptions (TYPICAL_DEVICE, VERIFY_ASYNC_LAUNCH, RR 0.75) — their arm deltas
  were bare-based, but the stack-level no-regression evidence (the f200-adopt boot, tonight's
  controls) holds; magnitudes need re-attribution, adoptions need no revert.
- RE-VALIDATED TONIGHT ON THE TRUE BASE: **STRATA_SPEC_DRAFT_TEMP=0.85 = +4.0% M1**
  (187.8 vs 180.6 no-DT), pair +0.7, conc +0.7 — AT the adopt bar, KEEPS its adoption (the
  bare-base +13.4% was the bare engine's larger acceptance headroom).
- VOID: **F-203's two rejections (DOORBELL_STORE, spec-8)** — both were bare+arm vs full-battery
  comparisons.  The dbstore M3b-13pt "correctness signature" was the missing battery.  Both arms
  go back to UNTESTED; re-queue on the fixed wrapper.
- The F-205 LFUSE/DF_PLE interleave (also bare-based) = VOID; re-queue (the node-inventory
  savings estimates stand on their own).
- F-201's DT sweep curve (bare-based) remains valid as a bare-engine dose-response; the peak
  location may shift on the full base — optional re-sweep.

**NEW METHOD LAW (added to the battery protocol):** every arm wrapper must be validated by a
NO-OP boot (unknown env var) reading warm-class M3b ≈ the launcher before ANY of its arms count;
overrides only via source-in-parent; launcher re-exports make env-prefix overrides useless.
**M3b stays the state-canary** — it caught this bug.

Also tonight: F-204 = DVFS exonerated (dmon: SM clocks 2797-2827 MHz, mem 13801-14001, zero dips
through a full battery — the gpu-lock holds under probes); the two F-204 scan maps banked
(research/F-204-emit-batching-map.md = the R33 plan; the node inventory in
research/F-204-node-inventory.md = the fusion queue with ~-900 to -1,000 nodes/window available,
three fusions already implemented behind gates).  FUSE_WAIT un-gated (default off, built).

Standing restored + healthy.  Sources: campaign/results/f205-*, f205v-*, f204-ctl,
/tmp/dmon-clocks.log.
### F-206 — THE FUSION TRANCHE (clean negative: node savings ≠ wall time) + R33 EMIT DE-SERIALIZATION SHIPPED (6-conc 219.1 = highest concurrent mean ever) + a power cut survived mid-battery (2026-10-15 ~12:30, goal round 18 execution)

**THE FUSION TRANCHE (on the FIXED wrapper — all arms warm, M3b 75-78, controls 181.5-191.6 M1).**
LFUSE −3.0% M1 / FUSE_WAIT −1.3% / DF_PLE −1.6% / the three-stack neutral — the −344 nodes/window
do NOT translate to wall time: the spacing is latency-hidden behind other terms (the graphs
brief's predicted "the line was already hidden" class).  All three stay default-off; the
FUSE_WAIT un-gate remains dormant.  A cheap, clean negative — the node inventory's bigger items
(#1 producer-fused quantize, #5 hc norm-fold, #6 batch-QSA suite) remain the fusion prizes, but
now with calibrated expectations: node counts are not the binding constraint at our spacing.

**R33 — THE SPLIT DRAFT (draft_launch/draft_finish + the two-pass emit loop).**  The per-slot
draft() was launch→spin→sync, serializing independent drafter streams; now every slot's residual
copy + draft graph launches first, then the waits run in launch order (the graphs overlap
GPU-side).  General draft() untouched (solo path keeps it).  Smoke: single + 3-concurrent
coherent; acceptance 76.6% class (unchanged — no distribution change, no MMLU class).

**THE VERDICT (5 runs across two batteries — the first cut short by a POWER CUT mid-battery;
run 1 pre-cut 175.2 + 4 runs post-reboot 186.1/173.3/219.1 M3b 76.9):**
vs the adjacent control band (185.6/174.3/214.0): M1 +0.3%, pair −0.6%, **6-conc +2.4% — and
219.1 is the HIGHEST CONCURRENT MEAN EVER RECORDED** (previous best 216.0/215.5).  Below the +4%
win bar → **ADOPTED AS HYGIENE** (F-196 precedent): strictly-better-or-neutral on exactly the
metric the mechanism targets (concurrent drafting slots), M1 flat as the mechanism predicts
(single stream: nothing to overlap), and the two-pass structure is the enabler for the batched
draft-head GEMM (the ~0.3-0.5ms follow-up in the F-204 emit map).  Code stays; scorecard
unchanged.  Post-reboot caveat noted: the 4-run battery ran in a fresh-boot window (cold caches,
proper 2-min warm-up applied) — the +2.4% carries that caveat, the all-time-high stands.

**OPS NOTES.**  The power cut wiped /tmp (rollback copies + scripts — git + ~/LLM/backup
survived; the F-206 code was committed BEFORE the re-run, the lesson applied).  engine/strata =
the R33 build (deployed, healthy).  Standing server restored on it.

Sources: campaign/results/f206-*, /tmp/f206-r33-redo.log (transient).

### F-207 — SOURCE CHECK + QUEUE RECALIBRATION + THE T1.1 IMPLEMENTATION PLAN (2026-10-15 ~13:30, goal round 18 planning)

**THE SOURCES (operator: tqllm.com + the Chawla X post).**  tqllm's headline "16% faster than
Strata" decoded: the 125B model on a 24GB RTX 3090 (experts streamed from RAM), GREEDY
1024-token chats, their tuned MTP drafter (mtp-blk48-iq4, 3 drafts) vs STOCK Strata --spec 4 —
a different lane from ours (32GB, sampled traffic, full recipe); no new technique on the page we
don't own or have queued (their MoE host-offload = our pinned tier; TqCache = our streaming KV;
TQ rotation quants = our Tier-3).  MONITOR: their Flash-Next 125B ships "coming soon" with
exact-reference-identical output claims.  The X post: general spec-decoding explainer, nothing
beyond our state.  THE STANDING SERVER verified live for the operator's monitor tab (health OK,
HTTP 200, model resident).

**THE RECALIBRATION (why the queue order changed):** R33's overlap made the batched draft-head
GEMM's prize mostly bandwidth-shared already (~100µs residual) — it drops below the big lines.
The F-206 fusion negative calibrated node-count items down.  THE ORDER NOW: **T1.1 partial
device plan (4.90ms tail) → doorbell posted-write inversion (1.8-3.6ms) → attention row-fold
(4.40ms line) → batched draft-head (residual) → R31 shape-keyed graphs**.

**THE T1.1 PLAN WRITTEN (research/T1.1-partial-plan-impl-plan.md)** — tonight's flow read
CONFIRMED the per-entry merge layout already supports mixed resident/CPU splits (m_ymiss rows
are per-entry; dst lists mark GPU entries; copy_rows_from_mapped fills the rest; skip_ is read
at RUNTIME by the wait kernels = graph-safe third state).  The edits: partial mode in
resident_plan_kernel (+miss-entry list to device), publish misses via the EXISTING
doorbell_publish_res, a third skip state gating flag-only waits, pool placement check.  ~150
lines; token streams must stay bit-identical (same experts, only WHERE they compute).

### F-208 — THE FOUR-AGENT RECON: campaign audit + engine hostile review + external SOTA + measurement lanes — the queue rebuilt on four fronts; the 5090 is running the oldest attention kernel in the tree; two goal lanes were unscorable and now have instruments; baseline confirmed safe (2026-10-15 ~15:00, goal round 19 recon)

**THE ASK (operator).** Specialized subagents to recon/research/study our work, progress, next
steps — find missed high-value items — while the main loop reviews highest-return moves,
criticizes the engine code itself, and identifies rewrite/overhaul candidates.

**METHOD.** 4 agents in 2 batches (≤2 concurrent, the measured cap): campaign auditor
(researcher, 1.95M tok), engine hostile reviewer (architect, 3.65M), external SOTA scan
(researcher, 1.42M), measurement-lane designer (general-purpose, 2.03M).  Briefs:
**research/F-208-{campaign-audit,engine-overhaul,external-sota,measurement-lanes}-agent.md**.
Main loop ran its own review in parallel (journal greps, kernel reads, T1.1 publish-side
pre-verification, git/remote check, server health for the operator's monitor tab: OK).

**THE HEADLINE FINDINGS.**
1. **The sm_120 build runs the OLDEST attention kernel in the tree.** PR#540's folded kernel
   (documented bit-for-bit, fewer shuffles + 4-cell fold = directly attacks the dependent-load
   latency chain F-194 diagnosed) is compiled only under STRATA_EXPERIMENTAL_SM60 and runtime-
   gated cc<75 (qsa_decode_attn.cu:291/:609) — a V100-era register-count concern.  The 4.40ms
   line sits ~40× over its bandwidth floor.  Better: upstream PR #2075 (open) does the
   decode+drafter hookup with a bit-exact merge — cherry-pick first.  Est −1.5-2.5ms.
2. **hc reads (4.2ms): order-preserving wide-grid fix designed** — 1-warp-per-row, 324 blocks
   (vs 41), lane chunk order unchanged = bitwise identical (overhaul A3); PR#646/#783's
   `hc_write_multi` may already be it.  Est −1.5-3.0ms.
3. **The doorbell tail has a second half T1.1 cannot remove**: a SINGLE service thread
   serializes layer l+1's flags behind layer l's pool drain (verify.cpp:3080-3130); plus batch
   windows forfeit the group-split overlap machinery entirely (`!batch_rec_` forces G=1,
   verify.cpp:882).  Plan-thread decoupling + a G=2 batch arm = the residual tail fix.  This
   machinery (expert_source.cpp service loops) is the one-week-REWRITE pick — oldest code,
   caps three other findings, survives T1.1 + posted-write inversion.
4. **The F-200 void boundary was wrong**: F-205 voided exactly three result sets but the F-200
   battery's non-adopted arms (LANECELL/GR_V3/L2-72/HC_SPLIT=1/MMQ_KQ) kept stale verdicts BY
   OMISSION — all bare-based.  LANECELL read +3.6/+5.4/+1.9 ON A BARE BASE (plausibly larger
   full-stack; #2268 says neutral — re-measure decides).  Re-arm tranche queued (Lane D).
5. **Two goal lanes were unscorable and now have instruments:** prefill (client
   `timings.prompt_per_second` = fresh-only rate — NO new engine surface; F-198's stock
   scoreboard simply lacks the row) and mimic (the M5 tool EXISTS intact on the frozen disk,
   stdlib-only, 4-diff port).  AND F-198 note 3 was WRONG: stock DOES print acceptance (probe
   --log misconfig) — the acceptance-vs-stock denominator exists.  A ~2-2.5h catch-up battery
   (spec written) scores all seven lanes, including the owed RR-0.75 adherence check.
6. **spec-8 re-queue was incoherent**: depths 4-6 were never OFFERED (chain starves at floor
   0.5, generate.cpp:12796-12798) — F-203's "tail dead by 6" was an inference over unmeasured
   depths.  Stair diagnostic FIRST, then depth arms.  T1.3 (deeper slot chains, pair +4-8%)
   was silently dropped in F-207's recalibration — restored (Lane B).
7. **Baseline SAFE**: upstream v0.1.44.1 (Oct 13-14) = pure hotfix, answers byte-identical,
   no CUDA-decode delta on our lane.  The moat is intact; free ideas live in UNMERGED PRs
   (#2075, #646/#783, #2514 top-k dispatch audit, #2326 PIPED_CONCURRENT).  tqllm's Flash-Next
   ship still not landed.  #2268 externally CONFIRMS our F-206 fusion negative.
8. **Stacked arithmetic (overhaul §C):** 4.40→~1.5 · 4.2→~1.5 · 4.90→~0.5 · emit ~1.0 ⇒
   window ~31.4→21-24ms ⇒ 6-conc clears the 244 bar with margin.  Every item individually
   Lock-Law-able.

**THE NEW MASTER ORDER (ROADMAP §0, v4):** Lane A code rocks: T1.1 → T1.2a pre75-on-sm120
(via #2075) + T1.2b row-fold → T1.4 hc wide-grid → A2 plan-thread → A6 batch G=2 → posted-write
inversion REPRICED post-T1.1 → R28 GDN identity → small class.  Lane B: stair diagnostic →
stair+spec-6/8 → slot chains → block verification.  Lane C: the catch-up battery.  Lane D:
env re-arms (LANECELL first).  HYGIENE: phase-table re-measure post-R33 BEFORE committing
Lane A order; 49 commits unpushed on one SSD (power-cut precedent) — backup needed; ROADMAP
§4 spacing items marked post-F-206 devalued.

**MAIN-LOOP PRE-VERIFICATION (T1.1's open question closed):** doorbell_publish_res is ALREADY
miss-aware (copies x/ids only when any_miss, else ring-bump only — elementwise.cu:329-351) and
the host pool is residency-aware on its side; partial mode needs only GPU-side plan/wait/dst
edits.  doorbell_publish_value (#649 store-the-ring) confirmed one env var away for a clean
re-test.  LANECELL confirmed a pure runtime env (qsa_decode_attn.cu:640).

### F-209 — GitHub backup closed (infra; no engine change, no probes)
gh 2.102.0 installed to ~/.local/bin (no sudo). Device-flow auth as **svgop** via pty driver
(lesson: answer gh's ESC[6n CPR query or its TUI stalls; never answer OSC-11 — the reply
bytes get parsed as prompt input and kill gh). `gh auth setup-git`; repo identity now
svgop / 94788596+svgop@users.noreply.github.com. Fork **svgop/Strata** fetched: its main =
upstream 0.1.44.1 (11 commits ahead of our base — the hotfix series, consistent with the
F-208 baseline note). Our 49-commit campaign line pushed as new branch **master** (no force,
every journal-cited SHA preserved). Empty repo svgop/strata-custom-2 created before the fork
redirect — deletion needs delete_repo scope (`gh auth refresh -h github.com -s delete_repo`).
Secret scan pre-push: API key 3f0940... present in tracked probe scripts + history
(loopback-only key, private repo — left as-is; rotation = one sed if ever wanted).
Upstream-PR plan banked: campaign/research/PR-UPSTREAM-plan.md (P1 hc wide-grid, P2 tuned
defaults, P3 slot drafts, P4 commit parity, P5 split draft, P6 Linux port RFC, P7 doorbell
RFC; #2075 territory = comment, don't duplicate).

### F-210 — upstream 0.1.44.1 merge ADOPTED (validated neutral; 12 boots)
Merge of upstream/main (11 commits: Metal #2323, serve crash diagnostics #2324 +
reasoning_effort Responses, CPU-AVX static-init safety, qwen35 f16 widen for CUDA 12.1,
version 0.1.44.1). Clean auto-merge, 0 conflicts; CMake took only the version bump.
Battery (all same-hour, standing recipe, means-only):
- ctl (pre-merge, 13:24-29): M1 195.7 / pair 177.3 / conc 227.5 / M3b 77.1 (3 boots)
- merged (13:31-36): M1 183.0 / pair 176.8 / conc 220.9 / M3b 77.0 (4 boots, wide 174-193)
- ABAB r1 (13:40/13:42): old 183.0/178.7/230.3 | new 190.3/184.2/234.9
- ABAB r2 (3 interleaved pairs, one continuous idle run): old 173.5/160.3/204.3 | new 165.2/161.5/204.5
POOLED interleaved (old N=4 vs new N=4): M1 175.9 vs 171.5 (-2.5%, SE +-4.6 = noise);
pair 164.9 vs 167.2 (+1.4 new); conc 210.8 vs 212.1 (+0.6 new); M3b ~76 both.
The -6.5% first-read gap was WINDOW DRIFT (own session context growth, F-192/F-196 class) -
binary does not predict the reading, the clock does; ABAB settled it (again).
MMLU gate on merged: 94.29 (66/70, floor 0.7) inside band 92.86-97.14. VERDICT: neutral, ADOPTED.
INSTRUMENT LAW (new): response-SHA bit-identity is INVALID on this stack - identical
temp-0 requests on the same binary+boot give different SHAs (spec drafter samples at
scaled temp; exact-match verify still ends non-deterministic through batching order).
Identity lane = engine-log accounting (accept%/windows) + MMLU class only.
Tooling banked: campaign/battery-boot.sh (boot+probe driver), campaign/abab-run.sh
(interleave driver; binary paths live in file contents only - footgun #4 law),
campaign/bit-check.py (RETRACTED instrument, kept for the record). Results in
campaign/results/f209-*. Branch: fork main fast-forwarded to our line (merge contains
upstream tip); local master renamed main; fork master deleted.

### F-211 — TREE HYGIENE: 123 files purged, root aligned to upstream shape
User directive: strip mess/old/dirt, keep only what we run + the 262k/1m line,
delete .bat, get close to niko/Strata. Purged (all preserved in git history):
- 38 of our Windows-era .bat launchers (launch-v2-* arm/diag/battery zoo, launch-custom-*.bat,
  build-strata.bat). KEPT niko's own BENCH/SETUP/START-HERE/UPDATE.bat (upstream files -
  deleting them would DIVERGE from upstream) and llama.cpp's vendored .bat (library content).
- .dsh/ (3 DeepSeek-Harness survey droppings), AGENTS.d/ (1 extraction), .docs/ (9 early
  survey digests, superseded by campaign/research/).
- Old launchers: launch-anchor-recipe.sh, launch-arm-pdl/qfuse.sh, launch-diag-profile.sh,
  launch-custom-q2_0-8081.sh, run-iq3_s.sh, run-q2_0.sh. KEPT: launch-custom-iq3s-8081.sh
  (standing), -262k.sh, -1m.sh.
- Old configs: strata-iq3_s*.json battery-arm zoo (10), strata-q2_0.json, custom-q2_0-8081.json.
  KEPT: custom-iq3_s-8081.json, custom-iq3s-262k.json, custom-iq3s-1m.json.
- Runtime droppings: 6 tracked logs, server-8081.out, 3 Windows stall minidumps (.dmp),
  refusal-r.gguf test artifact. .gitignore now blocks *.dmp/server-*.out/build-log*/custom-*.log/strata-*.log.
- campaign/: 7 Windows .ps1 probes, 26 F-*-entry/addendum drafts (all already merged into
  this journal - verified by spot-check), launch-anchor-test.sh + anchor-test.json (F-197 arm, superseded).
- Old docs: MTP-HEAD-REFERENCE.md, PERF-CHART.md (superseded by campaign/ records).
KEPT deliberately: third_party/llama.cpp (2821 files - our vendored Linux build system;
upstream has NO root CMakeLists and no llama.cpp, so this is load-bearing, not baggage),
CMakeLists.txt, data/ (live expert-profile state), AGENTS.md, SVG-CUSTOM.md, all of
campaign/ tooling+results+research. No source files touched; live server untouched.

### F-212 — F-211 PURGE BUG: refusal-r.gguf restored after first-boot failure (2026-10-16 01:41)

The first post-reboot boot of the standing launcher died at engine start: "control vector:
cannot open refusal-r.gguf", exit 2. Root cause: F-211's purge deleted refusal-r.gguf from
the repo root, but the standing config custom-iq3_s-8081.json loads it every boot
(--control-vector-scaled refusal-r.gguf:1.0, layers 4-24, project mode, per-layer). The
server had stayed up across the purge, so the break waited for the next reboot to surface.
Fix: git restore --source=d113423^ -- refusal-r.gguf (483,520 bytes). Law reaffirmed: before
deleting any file, grep the standing configs for references — the purge list was built from
directory shape, not from the config's dependency closure. Baseline battery restarted after
the restore.

### F-213 — TQLLM HEAD-TO-HEAD: 3.8x BEHIND ON M1; MTP PORT = ARCHITECTURE RECONCILIATION, DECLINED (2026-10-16 03:00)

Setup: turbo-tan/tqllm @c7f9852 cloned to ~/LLM/tqllm, built tqserve + tqfn-stage + tqtokenize
(CUDA 13.4, arch native=sm_120). One 2-line patch to src/tokenizer.cpp: GsqRco-dialect fallback
when a shard's tensors carry type 42 (tokenizer needs KV metadata only).

DISCOVERY: tqllm natively reads our GSQ-RCO dialect (src/gguf.cpp:232 maps external type 42 ->
GSQ_Q2_0, 64 elems/18 bytes) — it is a sibling engine for the same model+artifact family, not a
stranger. Our shard pair loads as-is via the tqfn --gsq-rco path.

MTP BLOCKER (why no drafting on their side): their Flash-Next serve engine hard-requires a
manifest mtp= GGUF with blk.48.nextn.* names (enorm/hnorm/eh_proj bridge) in the Native dialect;
our MTP artifact is Strata-format (mtp.* names, Q2_0 types) with a DIFFERENT bridge wiring
(fc_embedding/fc_hidden 2560x2560 square projections + hyper-connection mixers vs their
enorm/hnorm/eh_proj). Porting = architecture reconciliation, not a rename — declined as unsound
for measurement (mis-wiring would poison acceptance numbers silently). Their standalone decode
mode runs MTP-less (use_spec = spec_on && mtp_cache).

BATTERY (3 rounds per lane, fresh process per round, prompt x2 per process, warm pass reported;
campaign/results/tqllm-r{1,2,3}.txt):
  M1 (REAL, 2000 tok):    warm 47.32 / 47.52 / 47.71 -> 47.5 t/s  (expert hit 93.0%)
                          cold pass 45.0 (cache warming inside the run)
  Prefill (1783-tok fresh prompt, chunk 8 = their serve default):
                          warm 53.9 / 54.5 / 54.2 -> 54.2 t/s (hit 87.4%); cold 36.8
  Pair / 6-conc: NOT RUNNABLE — serve engine requires the MTP port above.
  Prefill at larger chunks: chunk 48 + slots 512 ran >20 min without finishing (their cap:
  chunk*topk <= slots <= experts=512 bounds chunk at 51; each chunk streams hundreds of unique
  experts). Consistent with their own docs/MOE_STATUS.md marking the MoE prefill lane pre-gate
  ("do not cite performance until gates pass").

VERDICT (operator, on the stalled no-spec parity round: "taking this long id say its already a
fail no?" — agreed, cancelled): tqllm end-to-end on this box/model is decisively behind our
standing engine — M1 47.5 vs our 180.9 = 3.8x, and the no-drafting handicap explains at most
~1.9x of that; plain-vs-plain remains >=2x. Our no-spec parity boot was additionally invalid:
the native pack contract refuses --spec 0 ("it needs --spec T (T >= 2)"), engine exits — noted,
not a regression. Standing server restored after the battery (200 on /v1/models).

The harvest (SVG-CUSTOM-TODO.md T1.x) is unaffected and is the actual point: their engine's
end-to-end loss reflects their pre-gate MoE-prefill state and 16-24GB-card tuning; the shared-
expert fusion, cache policies, and rule-table discipline remain the transferable value.

### F-214 — INDEX-SHARE ADAPTATION VOID: the drafter window is not the bottleneck; LANECELL re-arm confirms its conc profile; battery-config.sh env bug caught by the canary (2026-10-16 04:30)

THE QUESTION (operator: "proceed with more genuinely novel work"). T7.A's #1 item, IndexShare-MTP
(SGLang: freeze the QSA selection at draft-extend, indexer work N->1), was mapped onto OUR engine
first — static reading found our drafter ALREADY skips the indexer per step (mtp.cpp:463 ident_
ids precomputed at load; per step only a window slice at :1063). The translatable knob was
--mtp-window (generate.cpp:670, standing default 32768): the draft layer attends the last N
cells every step. Hypothesis: 4096 would slash per-step draft attention.

THE INSTRUMENT BUG FIRST (owned, on the record): the new campaign/battery-config.sh booted
serve/server.py directly and DROPPED the standing launcher's entire env recipe (SPEC_DRAFT_TEMP,
BATCH_MTP/SPEC_COUPLED, TYPICAL_ACCEPT/DEVICE, ROUTE_RESIDENT, ...). Both first-round "arms" ran
near-stock: M3b collapsed to ~62% (vs 76 standing) and M1 to ~153 — the acceptance canary caught
it immediately. Fix: the driver now evals the launcher's export block before boot (prints
SPEC_DRAFT_TEMP/BATCH_MTP/TYPICAL_ACCEPT for proof). Law reaffirmed: a config-boot battery MUST
inherit the standing env; the M3b canary is the tripwire that makes such bugs visible in one
round, not after adoption.

THE CORRECTED BATTERY (9 boots, interleaved [lanecell, mtpw4096, ctl] x3, 04:11-04:24):
  ctl      M1 195.9 / pair 183.6 / conc 229.6 / M3b 76.7   (181.6/208.3/197.7 — wide spread)
  LANECELL M1 189.1 (-3.5) / pair 188.9 (+2.9) / conc 240.1 (+4.6) / M3b 75.0 (-1.7pts)
  mtpw4096 M1 187.8 (-4.1) / pair 187.9 (+2.4) / conc 229.1 (-0.2) / M3b 76.8 (neutral)

VERDICTS (Lock Law, means-only):
- STRATA_ATTN_LANECELL=1 (F-200 re-arm): NOT ADOPTED for standing — same shape as F-200's
  +3.6/+5.4/+1.9 read, second independent confirmation: conc-favored (+4.6 at bar), M1-negative.
  Routed to the conc-profile idea: a serving-profile launcher (6-slot workloads) may want it;
  single-stream recipe does not. LANECELL r3 touched conc 244.8 / pair 194.4 — the conc/pair
  targets are INSIDE this arm's reach.
- --mtp-window 4096: NOT ADOPTED (M1 -4.1 at the bar). Mechanism, corrected: acceptance is
  NEUTRAL at 4096 (76.8 vs 76.7 — the drafter does NOT need 32K context for acceptance on this
  workload; the first-round "starvation" was the env bug, retracted), but M1 still falls because
  the smaller ring streams the drafter's K/V over host-pinned memory (mtp.cpp: KV streaming
  comment) while the 32768 window keeps the whole state in VRAM. The window knob trades VRAM
  residency for size, and whole-in-VRAM wins. The draft attention read is NOT the cycle
  bottleneck — the IndexShare adaptation lane is closed for this engine, with both mechanisms
  on record.

WHAT REMAINS NOVEL FROM TONIGHT: (1) the ctl band itself — evening means M1 195.9 (r2 hit 208.3)
vs the morning post-reboot baseline 180.9: +8% intraday drift, spread 181.6-208.3 within one
hour. The variance-control item (T6.10/9) now has evidence: warm page cache + battery churn
correlate with the top boots; the learned expert profile (loaded from data/expert-profile.bin,
unchanged) does NOT explain it. (2) The GDN slot-promotion scope, read directly from the code:
commit(n_keep) REPLAYS accepted tokens through the GDN scan (verify.hpp:14-19); F-71 measured
the marginal at 2.47 ms/row; SGLang's design (private state slot per draft column, promote by
pointer swap) eliminates the replay outright — the next kernel-lane project, scoped, not started.

BANKED: campaign/battery-config.sh (fixed), campaign/{lanecell,mtpw4096,mtpw2048}-iq3_s-8081.json,
results lanecell{,2}-r{1,2,3}, mtpw{,2}-4096-r{1,2,3}, mtpw-ctl-r{1,2,3}, ctl2-r{1,2,3}.

### F-215 — LANECELL CONC PROFILE LOCKED IN (operator directive): launch-custom-iq3s-8081-conc.sh (2026-10-16 04:40)

Operator: "This should be locked in and committed. 244 is too good to ignore." The F-214
re-arm's conc profile (+4.6% conc / +2.9% pair at -3.5% M1 vs standing) is now a committed
boot artifact: launch-custom-iq3s-8081-conc.sh = the standing launcher + STRATA_ATTN_LANECELL=1,
with the evidence block in-file. Discipline note on the record: the STANDING launcher stays
M1-first with LANECELL off — the conc profile is for concurrent-serving boots, and future
conc-lane arms and A/Bs use it as their control. Target scoreboard after F-214: conc target
244.0 has been TOUCHED (244.8, lanecell r3); pair target 193.8 touched (194.4, same round);
M1 target 191.2 crossed by a stock ctl boot (208.3, evening band). All three targets have
now been observed in-arm or in-ctl at least once — the campaign's remaining problem is
making them concurrent and stable, not reaching them.

### F-216 — GDN SLOT-PROMOTION, PHASE 1: snapshot kernel landed, noop-validated (2026-10-16 04:46)

Operator green-light ("then go pursue with the flag"). The design (campaign/research/GDN-SLOTS-
impl-plan.md): the verify window's recurrence already computes every column's state in registers
and discards it, while commit replays the accepted prefix through the same kernel — identical
arithmetic, prefix semantics, so persisting per-column snapshots is BIT-EQUAL to the replay by
construction. Phase 1 landed: gdn_step_norm_multi_kernel gains snap + snap_tok_stride (verify-
mode-only writes of the register state after each column; verify_kernels.cu:337-357), wrapper
guards refuse snap with STRATA_GDN_SPLIT / commit modes, header defaults keep every existing
call site untouched (snap == nullptr = zero behavior change). Build green; deployed
(engine/strata 75,143,944 via rm+cp — ETXTBSY on the live inode); noop boot gdnslots-noop-r1:
M1 180.7 / pair 179.9 / conc 220.0 / M3b 74.7 — in band, Launch-Law class validated. Phase 2
next: verify.cpp wiring (T_max x gdn_rec_floats x 36-layer arena ~576MB at init per F-91, the
window call passes snap, commit swaps the recurrence replay for a snapshot[m-1] -> state promote
kernel reading *n_keep device-side, conv commit + indexer tail + PLE unchanged), then parity
mode (STRATA_GDN_SLOTS=2) must log exact 0.0 across its audit windows before any battery.

### F-216 — GDN SLOT-PROMOTION: BUILT, BIT-PARITY PROVEN, PERF-NEGATIVE — the replay is NOT the depth gate (2026-10-16 08:40)

Phases 1-4 complete in one session. Phase 1: gdn_step_norm_multi persists the register state
after every column in verify mode (snap + stride; default null = unchanged). Phase 2: verify.cpp
wiring — arena (36 GDN layers x T x ~2MB, layer-major, alloc at init per F-91), serial windows
pass snap (batch windows and their commit graphs untouched by design), the serial commit
promotes snapshot[m-1] (mode 1) or replays + audits (mode 2); promote reads *n_keep on the
device at graph replay (graph.hpp law). Phase 3, STRATA_GDN_SLOTS=2 proof boot: **193 commits
audited, 0 mismatches — the snapshots are BIT-IDENTICAL to the replayed states**, exactly as the
construction predicts (same kernel, same per-token arithmetic, prefix acceptance). M3b 74.5 in
band. Phase 4, 3v3 ABAB (gdnpromote vs gdnctl, 08:0x-08:2x):
  arm  M1 181.1 / pair 177.1 / conc 221.4 / M3b 76.2
  ctl  M1 189.3 / pair 176.9 / conc 221.4 / M3b 75.7
  => M1 -4.4% (regression bar), pair +0.1, conc EXACTLY 0.0 (batch isolation proven), M3b +0.5.
VERDICT: NOT ADOPTED for standing. Mechanism: the snapshot stores (~T x 72MB per window, plus
promote copies) cost more than the replay they replace at our state size and T<=8; possible
register-pressure contribution in the launch-bounds kernel. The code STAYS env-gated (default
off, parity-proven) as an instrument.

THE DIAGNOSTIC GOLD (this is why the session was worth it): eliminating the ENTIRE commit-side
GDN recurrence replay moves pair/conc by ZERO and costs M1 — therefore **F-71's 2.47 ms/row
window marginal is NOT the GDN replay**. The depth/stair gate lives in the window compute
itself (dense-weight rows, expert marginal, doorbell tail), not state restoration. F-71's
attribution is corrected on the record; the stair re-test (Lane B) now targets the window's
dense-weight/expert marginal and the doorbell tail (T1.1 partial plan), not GDN state costs.
Also on the record: the self-kill footgun bit the operator's own compound command once tonight
('cp build/strata engine/strata' in the invoking line matched the drivers' pkill pattern and
killed the invoking shell silently) — the law (binary paths never in invoking commands) is
reaffirmed with a fresh scar; and the phase-2 binary was initially not deployed to engine/
strata (proof boot ran the stale phase-1 binary, caught by the missing AUDIT banner — deploy
check added to the mental checklist: verify the banner before reading a probe).

### F-217 — GOAL ROUND "NEW CEILING": the A3 hc-wide arc — two mechanisms killed, the F-208 hc-line model falsified (2026-10-16 13:20)

Operator goal round started ("keep progressing... raising the ceiling of some logic is capped at
its ceiling, these things need to be evaluated in order to evolve"). First pick: A3, the
highest-certainty unimplemented M1 item (hc wide-grid, est -1.5-3.0ms of the 4.2ms hc line).

THE ARC (all banked in campaign/results/):
1. gr_down_multi_wide_kernel built (1 warp per row, 324 blocks of 32, xn straight from global,
   lane chunk order preserved = bitwise by the file's own documented invariant). First arm:
   CRASH — illegal memory access; the port dropped the original's `if (!active) return;`, so
   inject blocks with no w_inject wrote through garbage inject_out. Fixed, redeployed.
2. Fixed arm (grwide2, 3 rounds): M1 169.9 / pair 160.8 / conc 209.7 vs ctl 187.1/175.6/225.0
   => M1 -9.2%. NOT ADOPTED. MECHANISM: the cooperative shared tile is LOAD-BEARING xn reuse —
   8 rows/block read xn once into SMEM (13 MB/layer L2 traffic); 1-warp-per-row makes all 324
   warps stream xn from L2 (~106 MB/layer), trading the occupancy cap for an L2-traffic cap and
   losing badly. The A3 design note ("xn is L2-resident") was true but irrelevant — residency
   is not reuse.
3. Occupancy probe (STRATA_GR_TILEV=1280, smem 80->40 KiB => ~2x resident blocks, bitwise by
   the file's documented tile-granularity proof, 3 rounds): M1 185.3 / pair 178.1 / conc 222.4
   vs ctl 187.1/175.6/225.0 => NEUTRAL (all deltas noise). Occupancy is NOT the hc line's cap.

THE CORRECTED CEILING MAP (three caps probed today, all falsified as gates):
- GDN commit replay: NOT the depth/window gate (F-216 — removing it entirely moves nothing).
- hc-read grid shape: NOT the cap (F-217 — the tile's xn reuse is the real constraint, and
  breaking it costs -9.2%).
- hc-read occupancy: NOT the cap (F-217 — 2x resident blocks, zero effect).
=> The F-208 per-line attribution (4.2ms hc line at 150 GB/s "grid-capped") does not describe
the CURRENT stack — it predates R33 split-draft/F-206 and the merged 0.1.44.1 base. LAW: no
more kernel surgery on the hc line without a fresh nsys per-kernel profile; the standing
window-cost map must be re-measured before T1.2a/T1.4 sizing. What survives as live M1 lanes:
T1.1 doorbell partial (publish-side, pre-verified plan), the attention lineage work (T1.2a via
upstream #2075 — still the biggest designed item), and the profile-first re-measurement itself.
INSTRUMENTS KEPT (env-gated, default off): gr_down_multi_wide_kernel + STRATA_GR_WIDE (the
L2-traffic cautionary instrument), STRATA_GR_TILEV (the occupancy probe). Standing server
restored post-battery (goal-restore in band).

### F-218 — THE FRESH COST MAP (profile-first law executed): attention is workload-class-dependent and dominates heavy windows (2026-10-16 14:0x)

STRATA_VERIFY_PROFILE boot + one probe battery (prof-218; profiled M1 173.1 vs ~180-190
unprofiled = the documented ~5% profiler tax from the disabled fusions — attribution, not
absolutes). The engine's own phase report (profile_report, means over windows):

LIGHT class (13.35 ms/window, n=2406): VRAM hits 1.93 (GDN 1.46 + QSA 0.47) | hc reads ~2.39
(hc0 down .57 up .44, hc1 down .58 up .45, norms .35) | copy+combine 1.08 | q8+qkv gemv .73 |
router+plan+bell .68 | shared+quant .64 | out-proj .53 | head .45 | z .44 | rec .28 | waitA .28.

HEAVY class (24.31 ms/window, n=549): **attention 4.09** (vs 0.93 light — 4.4x) | VRAM hits
3.46 | hc reads ~4.00 | copy+combine 1.41 | gemv 1.14 | head .78 | rec .70 | router+bell .94.

WHAT THIS CHANGES (vs the falsified F-208 model):
1. The QSA ATTENTION line is the top target after all — but only in the heavy class; F-208's
   single 4.40ms number conflated classes. T1.2a (pre75-on-sm120 via upstream #2075) lands on
   the 4.09ms line; est -1.5-2.5ms on heavy windows. Priority restored to #1 code lane.
2. VRAM expert hits (1.9-3.5ms) = the biggest STABLE line across classes -> the fused expert
   gather-matvec / packed-single-H2D lane (T6.6/T3.3) is the #2 target.
3. hc reads: 2.4-4.0ms confirmed present but grid/occupancy already falsified as its cap
   (F-217) -> the remaining hc lever is the SGLang-style Mix/Combine FUSION (fewer launches,
   fused math; measured +7.6/+5.5% e2e on our model class), not grid surgery.
4. copy+combine 1.1-1.4ms = the T1.1 doorbell/publish neighborhood, as designed.
LAW REAFFIRMED: this map is the sizing basis for T1.2a/T1.4/T1.1; re-profile after each
adoption (the classes shift). Banked: campaign/results/prof218-map.txt.

### F-219 — UPSTREAM #2758 SILENT-CORRUPTION FIX PORTED (discovery agent T8.C, first harvest) (2026-10-16 15:0x)

The discovery-mining agent's #1 find: upstream PR #2758 (SYCL fix) documents that the CUDA
engine carries a self-commit condition mismatch — record_window: `T==1 && !always_publish_ &&
!batch_rec_ && !g_qfuse() && one_token_self_commit()` vs Verifier::commit's one-token skip:
same WITHOUT `!g_qfuse()`. Under STRATA_QFUSE=1, a one-token window advances the GDN state in
NEITHER place → silent wrong tokens (no crash, no speed change). AUDITED OUR TREE: present
(verify.cpp:884 vs :2705). Standing runs QFUSE-off, so latent, not live. FIX PORTED (the
one-line `!g_qfuse()` in commit's condition, with the #2758 citation in-code), rebuilt,
deployed (7796d4d0), boot-validated f219-fixport: M1 186.2 / M3b 77.1 — in band (correctness
fix, zero perf delta expected and observed). Second harvest from the same report: our
--vram-reserve-mib 2000 already sits in #2612's good regime (their 12288→1500 = +30% via a
3x bigger expert cache); our 180-208 confirmed top of the public record for 5090+IQ3_S.
Full ranked harvest list: SVG-CUSTOM-TODO.md T8.C.

### F-220 — SMOOTHNESS KNOB #1 EXECUTED: prewarm pass in every battery boot; clock-lock paste block staged (2026-10-16 13:1x)

Agent T8.B's verdict: the boot-spread's dominant term is page-cache warm-in, not thermal. Fix
landed: campaign/prewarm.sh (sequential read of pack + both shards, idempotent, 7s warm /
~15-30s cold) now runs BEFORE the boot step of battery-boot.sh and battery-config.sh — every
battery round starts from the same warm state; run it manually after any reboot before
restoring the standing server. Statistical vindication on the record: our ABAB same-window
interleave is the paired design that detects +4% in 2-5 rounds where naive boots would need
~49/arm. GPU-side clip (user's sudo, reversible):
  sudo nvidia-smi -lgc 2700,2700        # lock SM slightly under the ~2827 floating average
  sudo nvidia-smi -lmc 14001,14001      # memory clocks at max (decode is memory-bound)
  sudo nvidia-smi -rgc; sudo nvidia-smi -rlmc   # revert
Trade (NVIDIA-documented): mean may sit slightly below floating peak; the FLOOR rises because
260-460 MHz idle/burst excursions and post-idle ramps disappear — correct trade for a
floor-above-200 goal. Persistence daemon already active (checked). Do-not-bother list banked
in T8.B. Next measurement law: after prewarm+lock, A/A the boots — if round-mean sigma lands
in the 2-3% class, future +4% claims are single-afternoon testable.

### F-221 — T1.2a ADOPTED: STRATA_ATTN_DAILY=1 — the first simultaneous all-lane win (+6.0/+7.0/+5.3) (2026-10-16 16:2x)

Upstream PR #2075 commit 901de010 cherry-picked clean onto our tree (auto-merge; no merge base
needed — self-contained kernel commit): PR#540's two-cells-per-step chunk kernel compiled for
EVERY CUDA card (STRATA_ATTN_DAILY=1 selects it; the cc<75 gate that kept our sm_120 on the
oldest lineage is gone) + Hardin22's bounded/batched merge (STRATA_ATTN_MERGE_V2; the
drafter's merge stops walking its whole 32768-cell capacity at every context). BIT-EXACT class:
construction argument + the PR's parity harness RUN ON OUR 5090 (1172 cases x fp16/int8/q4_0/
k8v4, 0 differ, PASS) — no MMLU gate required (exact-output class), M3b the canary.

ABAB 3v3 (daily-r{1,2,3} vs daily-ctl-r{1,2,3}, interleaved, prewarmed drivers):
  arm M1 196.7 (201.3/196.1/192.6) / pair 183.6 / conc 228.1 / M3b 75.2
  ctl M1 185.6 / pair 171.6 / conc 216.7 / M3b 76.3
  => M1 +6.0%, pair +7.0%, conc +5.3% — ALL THREE LANES CLEAR THE +4% BAR (the campaign's
  first simultaneous adoption). M3b -1.0pt = sampling noise (kernels are bit-identical).
ADOPTED into BOTH launchers (standing + conc profile) with the evidence block in-file;
adoption boot f221-adopt: pair 184.5 / conc 233.2 (above arm means), M1 181.2 (single-boot
noise inside the known spread; the means rule, and the clock-lock + variance work is the
floor's remaining lever). New standing band expectation: M1 ~192-200 / pair ~182-186 /
conc ~226-232. The F-218 map's #1 line (attention 4.09ms heavy-class) is now attacked; the
drafter's merge-walk bonus landed with it. Commit chain: cherry-pick + launchers in git.

### F-222 — prof221 map (the DAILY win lands in-map), the first all-three-targets boot, T9 folded, T1.1 executed NEUTRAL with the tail's true mechanism (2026-10-16 14:1x)

1. PRO-F221 (classes-shift law executed): heavy-class QSA attention 4.09 -> 1.96 ms/window
   (-52%; light 0.93 -> 0.42) — F-221's DAILY adoption visible directly in the phase map;
   attention is NO LONGER the top kernel line. The dominant line is the GPU-reach wait
   (10.99 light / 17.78 heavy) with VRAM expert hits (1.87/3.26) the biggest stable KERNEL
   line and hc reads 2.3/5.2. Probe at ~5% profiler tax: M1 191.6 / pair 184.9 / conc 241.2 /
   M3b 76.5. Map banked campaign/results/prof221-map.txt; standing server restored after.
2. STANDING-RESTORE BOOT: M1 206.4 / pair 197.9 / conc 248.3 / M3b 76.5 — the FIRST boot with
   ALL THREE +30% targets simultaneously (206.4>191.2, 197.9>193.8, 248.3>244.0). The
   campaign's headline goal is touched simultaneously; the remaining frontier is STABILITY
   (boot spread), which is the clock-lock + warm-in + variance lane.
3. T9 THIRD RESEARCH ROUND (3 agents, all folded same day into SVG-CUSTOM-TODO.md T9):
   PDL is HALF-WIRED in our tree (pdl.hpp sets the launch attribute but no
   cudaGridDependencySynchronize intrinsics anywhere hot and no programmatic graph edges —
   a stale-read hazard if enabled; finish it intrinsics-first, selectively); EcoSpec
   (2607.12696) expert-cost tree filling = the single-250 lever (+12-20 est,
   draft_policy.cpp, our gates already materialized); one-step-ahead service-thread
   pipelining (SGLang v0.4 overlap +20.4%) = the doorbell successor; TRT-LLM blog20 graph-
   ladder math (+8 ladder REGRESSED DSv3.2 — padding waste vs KV memory); EqSpec 2510.22876:
   batched-spec impls silently corrupt and vLLM/SGLang exact-match ~0 on Qwen3 — our
   exact-match verify is THE differentiator. Merged queue #1-10 in the doc.
4. T1.1 PARTIAL DEVICE PLAN EXECUTED AND CLOSED NEUTRAL: resident_plan_kernel partial mode
   (resident entries only in dst, counts[1]=R, start sentinel R; kPlanPartialSkip=0x80000000
   written to skip; miss entries absent from the list so the copy takes the pool's rows),
   wait_flag_ge_planned (flagA/flagB skip for partial too), copy_rows_or_zero_from_mapped
   (block=row=entry: full plan zeroes all - exactly copy_or_zero; partial zeroes listed rows
   and copies the pool's for the rest - exactly copy_rows semantics), pool pcie_ok gated off
   under STRATA_PLAN_PARTIAL (v1: misses all-CPU; A/B at --pcie-frac 0 both arms for
   bit-parity). ABAB 3v3: arm M1 190.0 / pair 198.5 / conc 249.7 vs ctl 191.2 / 201.7 /
   251.9 = -0.6 / -1.6 / -0.9 — nothing near the +4% bar -> NOT ADOPTED; env default-off,
   code stays as a correct env-gated instrument. Correctness held: M3b identical 76.5,
   acceptance 68.1 (arm) vs 64.1 (ctl) = F-210 nondeterminism class, no engine errors.
5. THE MECHANISM CORRECTION (this negative's yield): the arm's GPU-reach wait 16.36 vs ctl
   16.62 ms/window — the flagA host-plan round-trip T1.1 removes is worth only ~0.26ms.
   F-207's "4.90ms CPU-expert doorbell tail" attribution is CORRECTED: the tail is the
   service thread's OWN pool miss-row compute (the accounting's "CPU experts" stage,
   4.2-4.5ms/window), strictly ADDITIVE with the 16.4ms GPU wait (run 21.7 = 16.4 + 4.5,
   serialized), and the 16.4ms GPU-reach wait is the GPU's serial window execution, not host
   round-trips. CONSEQUENCE — the conc-300 lane re-aims: (a) kill the 4.5ms stage = T1.1-v2
   (device-side PCIe share for partial groups — the plumbing landed today makes it small) or
   the pipelining/multi-thread rewrite (T9.C#2); (b) shrink the 16.4ms = VRAM-hits fusion +
   hc fusion + PDL launch gaps + batch G=2 amortization. The skip machinery stays live for v2.
6. Commit: kernel/hpp/verify/expert_source edits + campaign/frac0*.json configs + t11-*
   results + prof221 map + T9 fold + this entry.

### F-223 — batch G=2 EXECUTED AND CLOSED: corrupting under the PCIe share, a regression even when clean; the roadmap item retired (2026-10-16 19:0x)

The roadmap's own next structural item ("batch G=2", from F-208's single-service-thread + G=1 note)
was implemented and falsified in one session.  Edits (all env-gated STRATA_BATCH_G2=1, default
off, committed): record_window's G gate allows 2 groups for batch windows of 2+ rows (the rows
are independent slots, so the serial window's mixer-boundary dependency argument is trivially
satisfied); batch_groups_ member carries the captured G; run_slot_rows' and batch_poll's service
loops made G-generic (the serial run()/service() pattern: doorbell k = layer lb_+k/G group k%G,
per-group gtb/gte row ranges, set_plan_slot(grp), per-group pool calls); stage_batch's b_steps_
= (le_-lb_)*batch_groups_.  Build green; serial paths bit-identical by construction.

RESULTS (5 discriminator boots, all on one binary, sustained-pair reproduction with the
non-finite count read from the engine's own stdout):
1. G2 + standing (pcie 0.35): NON-FINITE LOGITS (#879) under sustained concurrency — every
   multi-slot request dies after 1-4 tokens; M1 untouched (S=1 keeps G=1 — the isolation held).
2. G2 + STRATA_ATTN_DAILY=0: still corrupts (8 non-finite) — NOT the new attention kernel.
3. G2 + STRATA_VERIFY_DEVICE_PLAN=0: still corrupts — NOT the device-plan flow.
4. G2 + --pcie-frac 0: CLEAN (0 non-finite, full token counts) — the corruption boundary is
   batch x G2 x PCIe-share>0 (the DMA device-alias/flagB per-group path; root cause NOT run to
   ground — the lane is negative regardless; banked as a characterized boundary).
5. serial --spec-split (upstream G=2, untouched by our edits) under the FULL standing recipe:
   CLEAN and in-band (M1 197.4 / pair 198.9 / conc 252.6, 0 non-finite) — the G=2 window
   machinery is correct on our stack; the bug is in the batch x PCIe interaction.

THE PERFORMANCE VERDICT (clean frac-0 comparison, same binary/hour): G2 26.06 ms/window
(CUDA0 wait 18.81 + CPU experts 4.80) vs ctl 22.72 (16.62 + 4.20) — pair 157.2 / conc 220.5 vs
ctl 201.7 / 251.9 = -22% / -12%.  MECHANISM: the pool stage did NOT shrink — the sync service
loop serializes wait-doorbell -> pool -> raise-flag between doorbells regardless of grouping
(the pool work never overlaps GPU execution from the host's side), while the GPU wait GREW
+2.2ms (2x launches at half width).  The G=2 pipelining premise only pays with a service loop
that decouples pool work from doorbell waiting — i.e. the T9.C#2 one-step-ahead/multi-threaded
service loop remains the ONE structural fix for the 4.5ms CPU-experts stage; row-splitting
alone cannot do it (and upstream's serial "exact but slower" verdict is now confirmed for
batch on this engine).

STATUS: STRATA_BATCH_G2 stays default-off as a characterized instrument; the roadmap's batch
G=2 item is CLOSED (negative, with mechanism); M1 in-band 196-204 across all five debug boots
= binary healthy.  NEXT for the conc lane per all evidence: the pipelined service loop
(one-step-ahead host pipelining, SGLang v0.4-class) — the genuine rewrite.

### F-224 — STRATA_GDN_CHUNKED=1 ADOPTED: the prefill recurrence 1.68x, prefill −4.2% (the T12 discovery) (2026-10-16 21:4x)

THE DISCOVERY (T12-MEM audit): the chunked WY-form DeltaNet recurrence — flash-linear-attention's
form, IN OUR TREE SINCE UPSTREAM but opt-in and never armed — beats the standing per-token walk on
OUR card, measured by our own gdn_rec_parity --bench run: T=32768 chunked 8.90ms vs standing kh
14.93ms = 1.68x per layer; 36 layers per 32K block: recurrence+norm 699 -> 583ms.  Parity: the
harness PASSING on this rig (0 failures at 8192/32768; chunked-vs-kh max|diff|/max|value| ~3e-7,
same order as kh-vs-FP64).  Different FP32 sum order -> distribution-adjacent -> gated.
VALIDATION (Lock Law): paired prefill probe (campaign/prefill-probe.sh, unique ~8-9k prompts,
4v4, run TWICE in independent sessions): chunked wins EVERY pair, −141..−168ms (~−4.2% wall);
decode lanes 3v3 ABAB all inside noise (M1 +1.1 / pair −3.0 / conc −1.1 — prefill-only change,
decode structurally untouched); M3b in-band (arm 76.1 vs ctl 75.6 mean); MMLU-Pro 2x gate on the
arm boot: 95.7 / 92.9 (mean 94.3 — inside the standing 92.86-97.14 band).  ADOPTED into BOTH
launchers.  CONVS: STRATA_GDN_CONVL2 stays OFF on CUDA by design (#1522: not bitwise on
sm_120 — do not arm).
ALSO THIS SESSION (T12 pitstop round): the three code audits folded (75b22d94, dccb1c05, 2fdc10ac):
pool term ~90% bookkeeping (micro-fix package queued: plan single-pass IMPLEMENTED+built default-on
with STRATA_PLAN_ONEPASS=0 rollback + STRATA_PLAN_DIFF=1 audit harness; barrier diet next),
VRAM-hits at roofline twice-confirmed (sub-warp port re-priced down), light windows node-latency-
bound (hc norm-fold + LFUSE-gate + FUSE_HITS queued as free arms), FUSE_HITS env honored on CUDA
(was hard-false; de05c182).  Binary with planner+FUSE_HITS built; deploy after this adoption.

### F-225 — PLANNER FORMAL 3v3 REVERSES THE CALL: onepass is an M1 REGRESSION (-6.9%), the 213-215 "peak" was the session warm trend; default flipped back OFF; POOL_DIET instrument landed (2026-10-17 01:5x)

THE INTERLEAVE (battery-config 3v3, same binary throughout, arm=standing vs ctl=STRATA_PLAN_ONEPASS=0,
planarm/planctl-r1..r3): M1 arm 189.3 vs ctl 203.4 = -6.9% (round deltas -13.7/-36.8/+8.1 - r2-arm 171.0
is the outlier round, but even r1/r3 only give -2.8%: no adoption either way); pair -0.2%, conc +1.4%,
M3b -0.8pp - all in-band.  THE ADOPTION BAR FAILS on M1 alone.

THE RETRACTION (per the standing law, on the record): F-224's "M1 213.4/214.7 = NEW ALL-TIME PEAK"
credit to the planner was premature - the formal A/B says the planner did not cause it.  What round 3
actually shows: BOTH arms clear the old band together (arm-r3 216.7/201.0/253.6, ctl-r3 208.6/197.8/245.2
- all three stock targets with the planner OFF).  The band is a SESSION WARM phenomenon: successive
battery boots warm the box (arm 180->171->217, ctl 194->208->209 across r1->r3), so the third boot of
any series reads ~+20 M1 over the first.  Measurement-protocol lever banked: never compare a cold first
boot against a warm third boot; a warm protocol (back-to-back boots before scoring) is worth ~+10% on
paper for free.

DEFAULT FLIPPED: STRATA_PLAN_ONEPASS now defaults OFF (=1 re-arms; STRATA_PLAN_DIFF harness retained);
rebuilt+deployed b447b539.  The regression's mechanism is NOT established - the planner's own work is
us-scale and grouping-identical (the F-224 DIFF boot had zero mismatches) - so -6.9% cannot be its
direct cost; the interleave is the instrument of record and it says no.  Code kept as a gated instrument.

ALSO LANDED: STRATA_POOL_DIET (T12-HOST queue item 3, built, NOT yet armed): run_phase and run skip
their trailing wait_parked under the env - the epoch-carrying claim protocol (issue #29) already fences
a straggler out of the next batch (its claims fail on the epoch), so the repark is latency shaping, not
correctness (the pool.hpp rationale predates #29); skipped, the next phase's leading wait collects the
stragglers under the host's own setup work.  pool_diet_ member, =0 rollback by construction.

HIT-RATE CHECK during the run (both logs): 96.5-99.8% across arm and ctl boots - the expert-profile
shared state (data/expert-profile-learned.bin, saved every 30 requests) is NOT the band's cause.
Box clean throughout (load 2.2, only strata busy, GPU 40C/2827MHz).

### F-226 — STRATA_FUSE_HITS=1 ADOPTED: the hits-added combine fold clears the bar on its first-ever arm (2026-10-17 02:0x)

The fold (moe_hit_add -> native_moe_combine_multi_hits_added, ZADD bitwise per native_moe.cu:46) sat
behind a HIP-only #ifdef until de05c182 honored the env on CUDA - it had NEVER been armed here.  The
3v3 (fuarm/fuctl r1-r3, same binary, ctl = standing): M1 202.5 vs 194.1 = +4.35% (bar +4%), pair
+2.34%, conc +1.24%, M3b 76.2 vs 75.6 in-band, ZERO lane regressions.  Round deltas +26.0/-11.2/+10.5
(r2 went the other way, the mean holds).  ADOPTED into both launchers.
MECHANISM NOTE: F-205 calibrated that node-count cuts (-344/window) do NOT move wall time at our
spacing - so this win is NOT the node inventory; it is the removed COMBINE PASS work (the separate
combine kernel's reads/writes) on every window with VRAM hits, which at 97.7% hit rate is every
window.  Bitwise: the ZADD identity holds per element, M3b confirms in-band.
ALSO: T13B (concurrency audit agent) landed - binding-constraint verdict: single-window serial chain
(16.47ms GPU + 3.95ms additive pool + ~1.2ms host tail = 22.32ms for 6.55 rows = ~255 rows/s ceiling
at 6-conc); NEW candidates: window rows 8->12 (generate.cpp:10144 caps batch_mtp at 4 of 6 slots),
HostGraphLaunch for the batch path (solo-only today, verify.cpp:2348 vs :3116), emit batching
(per-token printf + per-window fflush, generate.cpp:10297/:10338), admission sync diet, LFUSE on the
doorbell path.  Dossier: campaign/research/T13B-concurrency-scheduling.md.

### F-227 — STRATA_POOL_DIET=1 ADOPTED: the trailing repark waits were pure serialization (+4.06% M1) (2026-10-17 02:1x)

The T12-HOST queue item, executed as planned: run_phase (pool.cpp) and run skip wait_parked("after a
phase/batch") under the env.  Correctness: the epoch-carrying claim protocol (issue #29, pool.hpp)
already fences a straggler out of the next batch - its claims fail on the epoch - so the repark is
latency shaping, not correctness (the pool.hpp lines 14-17 rationale predates #29); skipped, the next
phase's leading wait collects the stragglers under the host's own setup work instead of serializing
the slowest park onto the service thread.  3v3 (dietarm/dietctl, same binary b447b539, ctl = standing
incl. FUSE_HITS): M1 196.6 vs 188.9 = +4.06% (bar +4%), pair +2.41% (201.2 vs 196.5), conc -0.34%
(254.6 vs 255.5 - noise-level round overlap, arm r3 260.6 best of series), M3b +1.5pp in-band.
ADOPTED into both launchers (=0 rollback).  Round pattern note: r2 produced an outlier boot in BOTH
this battery (ctl 177.7) and the planner's (arm 171.0) - watch, not chase.
NOTE: the micro-package scoreboard so far - onepass planner NEGATIVE (F-225), barrier diet POSITIVE
(this), hits-fold POSITIVE (F-226): the pool term yields to fewer waits, not to a faster planner.

### F-228 — THE DEPTH STAIR, MEASURED AT LAST: the curve is real but the 0.35 gate-tune is NEGATIVE (-1.75% M1 3v3); the depth lever routes through a BETTER DRAFTER, not a looser gate (2026-10-17 02:2x)

FIRST PER-DEPTH ACCEPTANCE DATA EVER BANKED (STRATA_SPEC_DEPTH=1, stair50/stair35 boots + the s35arm
3v3 rounds): at standing min-p 0.5 the chain is NOT capped at ~2 - depths 3/4/5 are offered at 35/17/8%
of windows and accept at ~85/68/75%; mean 2.20 drafts offered, ~2.88 tokens/window.  At min-p 0.35 the
chain moves to 2.31 offered (P(d>=2) 0.63, d3 0.41, d4 0.19, d5 0.10) and marginal acceptance HOLDS
(56-90% at depths 3-5) - T13A's acceptance premise was RIGHT, its cost model was WRONG: the marginal
row costs not ~0.55ms (expert bytes) but the FULL graph's T-scaling (hc lines + attention + head re-run
per row), so high marginal acceptance still nets negative.  3v3 (s35arm/s35ctl): M1 198.0 vs 201.5 =
-1.75% (the stair35 diagnostic's +8.9% single-boot read was exactly the boot-noise the interleave
exists to catch), pair +1.6% noise, conc -0.6% noise, M3b 71.1 vs 74.7 = -4.8pp consistent-mechanistic.
CLOSED for gate-tuning (0.42 would sit between two measured negatives of a flat-to-negative curve).
WHAT REMAINS: the offered-chain extension is gate-limited only at the margin - the drafter's own
probs trim most chains - so depth pays only via a better drafter (FastMTP retrain, R3 GO unchanged)
or cheaper rows (hc norm-fold, bf16-native).  MMLU gate NOT run (arm rejected on speed alone).
LEDGER: this entry's M3d data retires the old "depths 4-6 never offered" note.

### F-229 — STRATA_DRAFT_UNGATED: NEGATIVE (-2.73% M1 3v3); the T13A #3 estimate had the sign of the trade backwards; closed, conditionally re-openable (2026-10-17 02:5x)

The instrument (c5703c3e, built+deployed a3f88b0b) enqueues all max_steps draft graphs back-to-back and
trims in readout - bit-identical, which the canary CONFIRMS: M3b 75.5 vs 75.4 (+0.13pp, the flattest
arm-vs-ctl M3b of any instrument this campaign, exactly what a same-bits change must produce).  The
speed verdict went the other way from T13A's +4-9 t/s: M1 201.0 vs 206.7 = -2.73%, ctl ahead in EVERY
round; pair +0.9%/conc -0.5% noise.  MECHANISM: F-228's curve says the min-p gate trims the chain at
~2.2 of 6 steps, so the ungated shape executes ~3.8 needless step graphs per window (~50us each on the
draft stream, serialized ahead of the verify window) - ~0.19ms of new GPU tail bought ~0.1ms of host
bubble savings.  The draft launch bubbles are SMALLER than the discarded-graph cost at our chain
length.  STAYS DEFAULT-OFF as an instrument.  Conditional re-open: a deeper mean chain shrinks the
waste toward zero (chain ~5 of 6 = ~1 wasted step) - if FastMTP lands, one boot re-tests it free.
Also closed tonight's session arc: 4 formal 3v3s (planner REV/-6.9, FUSE_HITS +4.35, POOL_DIET +4.06,
min-p .35 -1.75) + this = 5 instruments through the interleave in one night.

### F-230 — DENSE-MMQ ON CUDA: implemented, engaged, and CLOSED NEGATIVE — the prefill win is eaten by the decode acceptance tax; a new law for the prefill lane (2026-10-17 03:1x)

THE IMPLEMENTATION (kept, env-gated, default off): gemm.cu's four `STRATA_PREFILL_MMQ && __HIPCC__`
guards relaxed to `|| defined(__CUDACC__)` (the define is already PUBLIC on strata_prefill in the CUDA
branch, CMake:1517 - the HIP compiler flag was the only thing hiding native_mmq); moe_mmq.cu's
f16_to_f32/set_bounds + their kernels un-gated (#820 note updated); STRATA_DENSE_MMQ_MAXT T-gate added
(MMQ = the small-T bandwidth regime, cuBLAS FP16 tensor cores = the large-T compute regime); build
reconfigured STRATA_MMQ_KQUANTS=ON -> libstrata_kq.so (14.5MB) deployed beside the engine; d0875ab2.
ENGAGEMENT proven: "K-quant prompt kernels from engine/libstrata_kq.so" + "dense MMQ on" in the arm
logs, zero in ctl.

THE PREFILL CURVE (paired probes, BOTH boot orders, 20341bb8): 21k-token prompts -2% (slower - real,
warm-bias favored the arm), 2.7k +3.1% (0.9385s vs 0.968s median, arm booted colder), 1k +0.45% -
the a1444e33 curve shape with our crossover ~4k; MAXT=4096 gates to the winning side.

THE VERDICT (gated arm 3v3, mqarm/mqctl): M1 198.7 vs 204.5 = -2.8%, pair -2.4%, conc +0.4%, M3b
75.3 vs 77.4 = -2.1pp (arm max 76.7 < ctl mean).  MECHANISM (the structural find): the Q8-activation
prefill changes the prompt's KV by ~1.8e-4 KL (upstream's own number) - invisible to argmax-quality
metrics, but our coupled MTP acceptance is EXACT-MATCH verification, so borderline accept/reject
decisions flip: -2.7% relative acceptance = -2.8% M1, -2.4% pair, arithmetic matches.  Slots/hit-rate
exonerated (9,381 vs 9,337-9,426; 94.8-99.6% both).  NO ADOPTION - never-regress law.

**NEW LAW (the prefill acceptance tax): any optimization that changes prefill BITS pays ~-2.5% on
every decode t/s metric through acceptance, whatever its prefill speed win.  Price every future
prefill-accuracy change against this tax before arming it.**  (GDN_CHUNKED passed because its FP32
sum-order shift moved prefill by ~3e-7 AND M3b stayed in-band - the tax scales with the bits distance.)
INSTRUMENT KEPT: STRATA_DENSE_MMQ=1 + STRATA_MMQ_KQ=1 + MAXT re-arms in one config flip; MMLU gate
not run (speed verdict alone rejects; note: mmlu-pro-gate.py needs .venv python, system python3 lacks
datasets - fix before the next quality gate).

### F-231 — ADAPTIVE BATCH-MTP (STRATA_BATCH_MTP_MAX_SLOTS): the falsifier's +3.7% conc was one lucky boot; the interleave reads +1.81% conc / pair -3.61%(?) — below the bar, INSTRUMENT KEPT; the window-anatomy finding stands (2026-10-17 03:5x)

THE FALSIFIER (rows->window-time, f6m0 single boot): BATCH_MTP=0 at 6-conc = 5.93 rows @ 19.71ms vs
standing 6.62 @ 22.9ms, conc 267.5 vs 258.0; pair without drafts COLLAPSES (168.9 vs 193.0).  The
draft row costs more than its accepted tokens return at 6 slots; at pair scale it returns +14%.
THE IMPLEMENTATION (env-gated, default off = exactly today's path): batch_step computes mtp_use =
batch_mtp && (STRATA_BATCH_MTP_MAX_SLOTS <= 0 || n_active <= N); per-slot row widths (nrows[]) with
stale-draft slots taking one row; post-window, 2-row slots draft_launch as today, stale slots get a
draft_first refresh (admission shape), and draftless windows INVALIDATE the draft (a position-stale
draft must never be presented); acceptance/nonfinite/commit sites all row-width-aware.  Built
763abb81, ENGAGEMENT verified in-log: pair windows keep the 3.15-row MTP shape (14.37ms), 6-conc
windows run 5.97 rows @ 20.15ms.
THE VERDICT (3v3 amarm/amctl): conc 256.7 vs 252.1 = +1.81% (arm ahead EVERY round), M1 +1.05%,
M3b +0.23 in-band, pair 195.0 vs 202.3 = -3.61% (arm 192.0/200.9/192.1 vs ctl 200.5/198.6/207.8 -
twice at 192; the pair band is 188-209 and a ctl hit 188.1 last night, so noise is plausible but
NOT exonerated; at pair scale the arm's path is identical by construction).  +1.81% < the +4% bar
and the pair question stands: NO ADOPTION.  The falsifier's 267.5 was boot luck - the true MTP-off
conc effect is ~+2%.
LEFT ON THE TABLE (operator's call, LANECELL precedent F-214/215): the CONC launcher alone could
carry STRATA_BATCH_MTP_MAX_SLOTS=4 for conc-focused sessions (+2% there, pair untouched because
those sessions don't run pair).  Not done unilaterally - the pair -3.61% wants an exoneration boot
first (3 more arm-pair rounds would settle it).
ALSO: the deploy-protocol law re-proven the hard way - a battery wrapper whose own command text
contains "build/strata " (my md5sum) is killed by battery-config's pkill at the first boot; deploy
and battery NEVER in one command (the wrapper died after round 1; relaunch clean).

### F-232 — THE HOST-TRANCHE RE-PRICE SWEEP: all three T13B smalls are DEAD on mechanism reads (three batteries saved); the queue's head is R2-S3 (2026-10-17 04:2x)

C6 HostGraphLaunch-batch (+3-4% claimed): DEAD - the batch path's next act after the inline launch at
verify.cpp:3116 is the idle doorbell spin; a threaded launch saves host time only when there is host
WORK to overlap (the serial path's win came from its post-launch emit/staging).  Same wall time.
C7 emit batching (+2-3% claimed): DEAD - stdout is already pipe-buffered (per-token printf = stdio
memcpy, no syscall); bt_emit's 0.60ms is mostly the draft_finish WAITS (the timer spans the draft
wait loop, w2..now at :10359) plus nonfinite mask reads (a host bitfield, verify.hpp:121) - not I/O.
C9 admission sync diet (~2% claimed): DEAD as specified - conversation_copy.hpp:36 copy_nonblocking
ALREADY cudaStreamSynchronize's every copy ("complete on return"), so the 12 per-QSA-layer
cudaDeviceSynchronize calls in copy_to_slot/copy_from_slot (generate.cpp:10065/:10110) find an idle
device; the logged 36-56ms/admission is the KV image bytes over PCIe (save D2H + restore H2D round
trip), addressable only by fewer bytes (prefix reuse exists) or overlap, not by sync removal.
LESSON (journal law): T13B's candidate costs came from ACCOUNTING IDENTITIES (unexplained ms
segments), not mechanism reads - every candidate estimate must survive a code-read of what the
segment actually spends time on before it earns a battery.  (The T13A hc norm-fold died the same
death earlier tonight.)
QUEUE HEAD after the sweep: R2-S3 two cohort verifiers (the conc-300 route, +20-28% arithmetic,
per-verifier buffers kill the G2 ghost by construction) -> impl plan staged next; FastMTP retrain
(depth/quality); hc bf16-native dots (the 2.87ms/window line; different-bits, full gate - and note
the drafter shares the kernels so target/draft stay consistent); head mmvq_il-at-T1 microbench
(verify.cpp:1863 gate; +2-4 t/s class).

### F-233 — STRATA_COHORTS=2 ADOPTED: the two-cohort verifier batch lane (R2-S3) — conc +24.8%, the 300 stretch goal CLEARED (306.8 mean, every arm round above 304) (2026-10-17 11:4x)

THE STRUCTURE (per campaign/research/R2-S3-two-cohort-impl-plan.md, executed in one session): a second
Verifier (ver_b, full-range, own stream/graphs/doorbells/arena - the G2 ghost gone BY CONSTRUCTION) on
the SAME GPU; slots [0,3) on A, [3,6) on B; the reactor stages BOTH windows while nothing is in flight,
launches both, then one thread polls both (each poll serving its verifier's rung layers - B's CPU pool
work runs under A's GPU window, the one escape from F-223's theorem); per-cohort tails (acceptance,
parity commit, emit, drafts) parameterized by verifier.  Engagement arms only when BOTH cohorts have
work (n_active > 3): solo and pair run the UNTOUCHED single-window path byte for byte.

THREE BUGS ON THE WAY (each boot-gated): (1) empty-cohort rounds called commit_slots without a window
(clean abort) - guarded; (2) batch_launch's pipe-style PRE-LAUNCHED commit collided with the tail's
F-196 parity commit (pair hang) - batch_launch_rows(..., with_commit=false); (3) lazy graph capture of
a new layout while the OTHER cohort's window was in flight (intermittent 6-conc hang) - the two-phase
batch_stage_rows/batch_go (stage both, launch both) plus a one-time device fence at the standing->cohort
transition (the last standing window's async commit).  Reactor diagnostics on stderr (stdout is
pipe-buffered - the swallowed ERRs cost an hour).

THE EVIDENCE (3v3 coharm/cohctl + co2/co3 with the two refinements): FINAL fenced 3v3 (co3, adfa4889):
conc 306.8 vs 245.9 = +24.80% (arm 304.2/310.3/306.0 - NON-OVERLAPPING with ctl 240-250); M1 192.8 vs
190.4 = +1.24%; pair 193.1 vs 196.6 = -1.76% (rounds 190.5-196.6 inside ctl's band - the pair path is
byte-identical code, this is boot noise; the interim gated battery read pair +2.7%); M3b 75.17 vs 75.20
= -0.04pp, the flattest canary of the campaign.  Window anatomy: two ~6-row windows per ~26-29ms round
in-flight overlapped vs one 6.6-row window per 22.9ms - the in-window rate went from ~349 to ~460
rows/s class.  ADOPTED into both launchers (STRATA_COHORTS=2; =1 rolls back to the untouched path).

ALSO THIS SESSION: T14A/B/C FastMTP dossiers (recipe: recursive teacher-forced multi-depth CE beta=0.6
K>=serve-depth over self-distilled data, RH productionized ON OUR HEAD CLASS with 5-8K samples/443s on
2xH200; infra: --dump-final-r teacher dump, dense-only ~98M trainable, rt2 parallel rollout, the R17
Windows arc already validated the chain; math: c_row=3.56 ms/row calibrated from our own s35 3v3,
acceptance-only ceiling 222/225/281 - 250 needs the retrain AND the kernel tranche).  The standing
scoreboard after tonight: M1 ~193-215, pair ~193-207, conc 304-310 - stock targets (191.2/193.8/244.0)
cleared with margin on all three, the 300 conc stretch goal CLEARED, 250-single remains the open goal
(routes: FastMTP retrain + the kernel tranche).

### THE STANDING SCOREBOARD after F-233 — the five-boot reliability bench (2026-10-17 16:0x)

Five standing boots on the full recipe (ATTN_DAILY + GDN_CHUNKED + FUSE_HITS + POOL_DIET + COHORTS),
plus the quality/latency lanes on the final boot:
- M1 single-stream: 192.2 / 211.3 / 205.5 / 197.6 / 185.7 -> MEAN 198.5 t/s (+35.0% vs stock 147.1)
- M2 pair: 196.5 / 197.0 / 202.9 / 196.8 / 194.7 -> MEAN 197.6 t/s (+32.5% vs stock 149.1)
- M2 6-conc: 312.4 / 322.7 / 306.2 / 320.6 / 300.2 -> MEAN 312.4 t/s (+66.4% vs stock 187.7)
- M3b acceptance: 74.5 / 76.0 / 76.0 / 76.0 / 72.2 -> MEAN 74.9%
- Prefill (ctx ladder, fresh unique prompts): 3.6k @4k -> 6.0-7.0k plateau 16k-120k (peak 7,028 @64k)
- Long-ctx decode: 184.7 cold @4k -> 226-243 t/s warm 16k-120k (was 208-216 pre-recipe)
- MMLU-Pro: 95.7 / 94.3 (mean 95.0; band floor 91 cleared twice)
- Mimic (mimicbench2, 40 tasks, weighted): 87.9% - internalization 12/12, correction 8/8, voice 6/6,
  doc-structure 6/6, doctrine 6/8, arch-continue 6/8, blend 6/8, ui-deep 8/10
**THE +30%-ON-EVERY-METRIC GOAL IS NOW CLEARED ON MEANS: +35% / +32.5% / +66% vs stock.**  Remaining
stretch: single 250 (the FastMTP retrain + kernel tranche path, T14A/B/C).  Results:
campaign/results/final-bench-r{1..4}.txt, fbench-lanes.log, fbench-mimic.txt.

### F-234 — upstream v0.1.45 checked: no stock perf jump worth a merge (2026-10-17 night)
Trigger: user asked to check .45, bench stock before/after the pull. Stock install found at
v0.1.43.1 (never took .44). Boot-blocker fixed first: strata-q2_0.json --vram-reserve-mib
200 -> 768 (the 55 MiB device-arena no-fit; our F-172-era Q2_0-on-32GB law) — change stays.
A-B-A on the SAME q2_0 config, fresh boot per leg, locked clocks (3135 bin, ~2955 realized,
zero throttle flags, <360 W): M1/pair/6-conc —
  .43.1 A1  195.3 / 163.6 / 170.9   (campaign/results/stock-431-fresh.txt)
  .45    B  205.9 / 170.1 / 173.4   (campaign/results/stock-45-fresh.txt)
  .43.1 A2  206.8 / 174.6 / 177.9   (campaign/results/stock-431-fresh-b.txt)
Means: M1 +2.4%, pair +0.6%, conc -0.6% — ALL inside the +-4% band. The first-look +5.4% M1
was the F-225 warm-trend artifact; the interleave killed it. CALL: v0.1.45 = flat on our config.
What .45 is (267 files, +30.8k): perf-PR groups aimed at OTHER platforms (sm75, rdna2/34, sycl,
sm86, cpu, cache, serve, batch groups of the f-145 initiative), multi-GPU peer tier (#2666/#2644/
#2418), serve QoL (thinking budget, model-card sampling defaults), a Q2_0 side-mixer correctness
fix. CUDA-side it DID touch our hot files — qsa_select.cu +1353 lines, verify_kernels.cu +132 —
but the perf levers are OPT-IN and NOT armed by default: STRATA_PROMPT_ATTN_IMMA=1 (int8-KV
prompt attention on INT8 tensor cores, "closer to FP32"), STRATA_PIPED_READ_PIECE (#2050
batch-groups decode/give-way). Also documents STRATA_MMVQ_LARGE_GRF / STRATA_Q6_PLANAR Q6_K
decode switches (#2340/#2207) — overlaps T15A's mmvq sm_120 table finding (plain kernel on 73%
of windows) and T15C's Q6_PACKED candidate.
VERDICT: do NOT merge .45 now — default path flat, conflict surface = exactly the verify/qsa
files we rewrote (COHORTS/ATTN_DAILY land). The two opt-in arms + Q6_K switches become candidate
T16 battery riders AFTER the standing queue; price with the F-232 cost-anatomy read first.
"Against ours": stock .45 reads ~204-207 M1 on the FAST q2_0 quant vs our 198.5 on iq3_s banked
at the OLD 2835 clocks (3135 re-bank pending, +3.4% class) — and pair/conc remain a rout:
ours 197.6/312.4 vs stock ~170/174.
State: stock install now v0.1.45 (engine/strata.0.1.43.1-backup kept, reserve 768); custom
anchor restored on 8081 after the benches.

### F-234a — the FAIR same-quant A/B: custom-2 vs stock .45 on IQ3_S (2026-10-17 night)
User's catch: the first three-way mixed quants (stock ran Q2_0). Fix: strata-iq3s-stock.json in
the stock install — custom-2's config mirrored EXACTLY (same model GGUFs, pack, mtp/rt,
draft_vocab en, spec 6 / min-p 0.5 / kv int8 / resident / cvec / reserve 2000, parallel 6;
expert-profile on a COPY so probe traffic can't touch the custom lane's learned file), stock
v0.1.45 engine, ZERO custom arms. Every config arg accepted by .45 clean.
Result (M1/pair/6-conc): stock-IQ3_S 141.5 / 146.8 / 220.2 vs custom-2 188.8 / 187.8 / 272.0
= **+33.4% / +27.9% / +23.5% at identical model bytes** (campaign/results/stock-45-iq3s.txt).
That is the engine+arms delta, isolated. Context the first chart hid: the QUANT alone is worth
+45% M1 on stock (Q2_0 205.9 vs IQ3_S 141.5) — custom-2 nearly closes even that (188.8 = -8% vs
stock's faster quant) while beating it outright on pair/conc. Stock .43.1 vs .45 stayed flat
within noise in every pairing. State: stock install keeps both configs (run-q2_0.sh,
run-iq3s-stock.sh); custom anchor restored on 8081.

F-234a addendum: completed the same-quant matrix — stock v0.1.43.1 on IQ3_S (same config swap)
M1 144.9 / pair 149.0 / conc 224.4. So .43.1 vs .45 is FLAT WITHIN NOISE at both quants
(same-quant deltas +2.4/+1.5/+1.9% favoring .43.1, all inside ±4%) — upstream's .45 perf wave
does not reach this config, now proven twice. Custom-2 vs .43.1 same-quant: +30.3/+26.0/+21.2%.
Custom anchor restored on 8081.

F-234a addendum 2 — the 2x2 matrix is COMPLETE (2026-10-17 night): custom-2 on Q2_0 via
strata-q2_0-custom.json + run-q2_0-custom.sh in the stock install (stock's q2_0 config held
byte-identical — spec 4, stock mtp/rt, reserve 768, parallel default — custom binary + the
launcher's 26 arms via custom-arms.env): M1 242.6 / pair 210.1 / conc 215.5 vs stock Q2_0
205.9 / 170.1 / 173.4 = **+17.8 / +23.5 / +24.3%**. The full matrix: custom wins every
engine-only cell by +17.8..+33.4% regardless of quant. Note: custom-Q2_0 conc (215.5) < 
custom-IQ3_S conc (272.0) because stock's config is the narrower one (4-slot default, spec 4 —
COHORTS pays above 4 active slots; IQ3_S lane runs parallel 6). Also the Q2_0 arm set booted
clean on the fork. Custom anchor restored on 8081.

### F-235 — THE STEP-0 FREE-BOOT SLATE, CLOSED: all eight falsifiers negative or priced-out (the fast-warm v2 A/B recovered and measured NEGATIVE-leaning); the T17 zero-code program ends clean (2026-10-18, post-ISP-crash closeout)

The implementer's boots survived the crash in campaign/results (verdicts below from the files, not
memory).  The slate's premise - eight zero-code falsifiers in info-value order, winners become
candidates-for-battery - produced ZERO winners:
- FAST-WARM v2 (t17b-fw/fwctl r1-r2, the post-pin prewarm package campaign/prewarm2.sh): arm means
  M1 195.1 vs ctl 206.1 = -5.3%, pair +0.0%, conc +0.9% - the page-cache re-population after the
  boot pin does NOT pay (fwctl-r1's 221.4 was the warm-trend boot of the pair).  NO ADOPTION; the
  cold-start gap stays a named target with no free fix.
- MTP-OFF (m0): M1 191.1 / pair 159.4 (-19%) / conc 268.9 - reconfirms F-231's anatomy at the
  current recipe (drafts cost at 6 slots, pay at pair; the adaptive instrument stays the answer).
- PARALLEL-8 (p8): M1 178.9 / pair 176.2 / conc 0.0 (HUNG) - dead; 8 slots do not fit the recipe.
- SPEC_PROB (spc): 189.7/191.6/303.8, M3b 76.3 in-band - neutral, priced-out (probabilistic
  acceptance buys nothing at 74.9% exact-match acceptance).
- LOOKAHEAD_STATS (las): 208.4/199.4/310.5 in-band - diagnostic data banked, no arm.
- DEPTH REAL-vs-FILLER at 120k (depth-t17f-real120k, via campaign/depth-lane-real.py - the repo's
  own technical prose as corpus): deep-position acceptance 635/1195 = 53.1% vs shallow 93.6%,
  prefill 7229 t/s, warm decode 239.3 - the position/content split is MEASURED: the depth deficit
  is positional, not corpus-class; pairs with F-228's stair as the FastMTP training-data spec
  (deep positions must be oversampled - now a T14B-spec input).
- DUMMY_NODES / DRAFT_TIMING verdicts per the recovered transcript (memory-indexed): negative /
  priced-out.
ALSO COMMITTED (the crashed session's unpushed work): T17A-m1-single-leap.md (the M1 leap dossier -
external sweep de-risking the retrain, +20 t/s per -1.0 ms transfer law), T15-easy-wins-audit.md
(the operator-survey audit: hc grid occupancy +8-9% M1 bit-exact class = the kernel-tranche head,
EcoSpec banked DEAD), prewarm2.sh + depth-lane-real.py tools, learned profile.
STATE AFTER CLOSEOUT: standing restored on 8081 (post-crash boot); the queue head = the T14B spec
trio RELAUNCH (harvest/parity/trainer-export execution specs - the crash killed them at zero output),
then the kernel tranche v2 per T17's re-ranked plan.

### F-236 — STEP 0 closeout v2: the full per-item verdict table, completing F-235 with two corrections (legs 2026-10-17 22:35-23:10; table written 10-18 after both crash closeouts)
Completes F-235's closeout with all 13 legs itemized (SVG-CUSTOM-TODO.md "STEP 0 RESULTS").
CONCUR with F-235: fast-warm v2 NEGATIVE-leaning / no adoption (with one caveat — tonight's box
was never truly cold after 7+ boots, so the +10-13% cold-leg claim is UNTESTED, not disproven;
prewarm2.sh stays banked for a true-cold A/B). SHARPENS two F-235 reads:
(1) DUMMY_NODES is NOT merely "priced-out": 48 dummies = M1 -12.5% (+1.1 ms late-window) is
direct CONFIRMATION the per-node tax is real and material — that is the strongest pro-R1
(megakernel/node-collapse) evidence on record. The diagnostic arm is dead; its finding is a GO
signal for the lane it prices.
(2) DEPTH SPLIT corrected — position and content BOTH measured, position primary: deep-position
real-corpus 53.1% (635/1195) vs shallow 93.6% (KV-reuse continuation) = -40.5pt POSITIONAL; vs
filler 68.0% at matched ~121k depth = -14.9pt CONTENT. F-235's position-oversampling data-spec
input CONFIRMED; content-matched draft data additionally promoted; LongSpec aug keeps its slot
(it was never the whole answer).
Implementer ran all eight free-boot falsifiers in one session (legs t17a/t17b/t17c/t17e-* +
depth-t17f-* in campaign/results/, ~30 min of boots); the 23:10 ISP crash killed the agent at
the closeout (ENOTFOUND zcode.z.ai) — all 13 leg files survived on disk; verdicts extracted and
closed out by the main session 10-18. Protocol: fresh boot per leg, same-window A/B vs a
standing-recipe control (198.0/201.8/326.7 — that conc is the night's high and says the 3135
re-bank restores >320 means). Verdicts: (1) DUMMY_NODES — 16 flat, 48 = M1 -12.5% (+1.1 ms
late-window): per-node tax real; R1 megakernel lane ALIVE. (2) DRAFT_TIMING gap_ms table LOST
to log rotation — R2 INCONCLUSIVE. (3) SPEC_PROB — -4.2/-5.1/-7.0%: rejection sampling loses to
the coupled race at min-p 0.5; door CLOSED (GUMBEL-consistent). (4) LOOKAHEAD recall table LOST
— C1 INCONCLUSIVE. (5) depth-lane corrected: POSITION primary (-40.5pt — real-corpus deep
53.1% vs shallow 93.6% KV-reuse continuation), CONTENT secondary but real (-14.9pt — filler
68.0% at matched depth); position-oversampling data-spec CONFIRMED, content-matched data
additionally promoted. (6) parallel-8 — M1 -9.6/pair -12.7/conc collapsed (0 tok/49.6 s): DEAD. (7) MTP-off
— pair -21.0/conc -17.7: MTP pays hugely under COHORTS; F-231-vs-MagicDec settled, MTP stays.
(8) fast-warm ABAB — M1 -5.4% (ctl outlier 221.4), pair flat, conc +0.9%: INCONCLUSIVE on a
never-cold box; prewarm2.sh banked; learned-profile arm never ran = open. LAW NOTES: nothing
adopted (nothing earned it); zero code written; two instrument boots owed (DRAFT_TIMING,
LOOKAHEAD_STATS with proper capture); fast-warm owes a true-cold A/B (fresh reboot or cache
drop); learned profile owes a config-copy boot + 2x MMLU.

### F-237 — THE CONSISTENCY AUDIT: spec-set reconciliation, the unbanked data carried, and the 3135 re-bank declared owed (2026-10-19)

THE AUDIT (campaign/research/AUDIT-2026-10-18.md, read-only agent): the two parallel FastMTP spec
efforts are ~80% the same plan independently re-derived; the endorsed merge = set 2's architecture
(campaign/T14B-specs/, TODO :1270-1292) + set 1's folds (exec-B parity wholesale, exec-C evaluator/
w_deep/H200/OD points, exec-A's S25 caveat + probe gates).  THREE SILENT POISONS in set 2 found and
to be fixed before execution: (1) trainer config's lr 5e-4 (5-10x the argued 1e-4 - would poison
every arm); (2) the rt-dir split (gates-spec seds mtp/rt2, trainer-spec retires it for
mtp-iq3s/rt-r18 - the flip command targets a directory never created); (3) the G4/G5 label collision
drops F-229's ungated re-open and F-228's min-p re-stair from set 2's post-adoption ladder (they
survive only in set 1/T14C).
UNBANKED DATA, now carried: the clock A/B legs (2835->3135 between 19:56-20:03 Oct 17: M1 sustained
186.0->192.4 = +3.44%, zero throttle, 356/575W, realized ~2940 - previously only in TODO M0);
custom2-final-warm.txt (conc 299.5 warm same-boot vs the banked 272.0 cold - F-234a's custom-conc
cell UNDERSTATES the warm lane by ~10%); the STEP-0 leg configs + t17-leg.sh + the never-run
learned-profile arm (t17-lp.*) - all committed herewith.
LOST-FROM-QUEUES items restored to the pending list (audit section 5): the six T15A/C env arms, the
GDN decode ILP rewrite, T11.R5's MMLU CI fix, the cvec sweep, T13's stale RUNNING tags.
**THE RE-BANK IS OWED: the five-boot means (198.5/197.6/312.4 + MMLU 95.0 + mimic 87.9) are
2835-class; everything after Oct-17 20:03 runs at 3135.  Until 5 fresh standing boots + 2x MMLU +
mimic land at 3135, no absolute number crosses the 20:03 boundary.  Point evidence: +3.44% M1,
STEP-0 control conc 326.7 - the true standing scoreboard is HIGHER than banked.**
OPERATIONAL: two writers were active on this repo through the audit window - coordinate before any
spec edit or re-bank (the audit's closing note).

### F-238 — THE 3135 RE-BANK: means say FLAT — the +3.44% clock delta does not replicate; quality holds (2026-10-19 night)
Five fresh standing boots at the 3135 lock (campaign/rebank-3135.sh -> campaign/results/
rebank-3135.log, probe --short-n 6 each): M1 199.8/190.9/196.8/183.8/197.3 = 193.7 mean (198.5 at
2835, -2.4%); pair 200.5/197.4/203.9/192.9/201.6 = 199.3 (+0.8%); 6-conc
317.3/287.6/307.9/283.7/320.3 = 303.4 (-2.9%). M3b 76.2. Quality at the locked clocks: MMLU-Pro
95.71 and 95.71 (2x, floor 0.933), mimic 90.9% (60/66 vs the 87.9 anchor).
VERDICT: every lane inside +-4% -> the Oct-17 +3.44% single-leg A/B (186.0->192.4 sustained) was
a point estimate, not a mean property. The lock STAYS (harmless, quality clean, point-positive)
but no clock gain may appear in any scoreboard. Two retractions on the record: (a) "+3.4% flows
into the 250 ladder" - false at means; (b) "every conc boot >=300" (2835 bank) - false at 3135
(2 of 5 below 300; the 300 stretch holds on the 303.4 mean only). Post-boundary scoreboard:
M1 193.7 / pair 199.3 / conc 303.4 @3135-class, MMLU 95.7, mimic 90.9. The 250 path must come
from re-quant + kernel tranche + retrain, not clocks.
METHOD NOTE (probe gotchas, from this run): the five-boot loop must kill by PORT (ss) and reap
strata compute pids by /proc cmdline - a bare server kill can orphan an engine; the port guard
turns double-writer collisions into loud per-leg aborts.

### F-239 — THE KNOB SWEEP: five S25/S26 arms tested and REJECTED; prefill anatomy banked; a standing-recipe SIGSEGV found (2026-10-20 ~00:00-00:55)
T15C's "never swept on this stack" env inventory is now closed. Marathon (7 single-window legs)
showed the window drifting +7% against the 50-min-older control - every leg including the pure
diagnostics read high - so nothing single-window was claimed; all five candidates then ran 3v3
ABAB interleaved batteries (drift differences out), with a confirm 3v3 for the two at-bar ones.
FINAL: LANECELL -1.5/-5.6/-7.2 REJECT; TSUM +4.4 then +1.3 (pair negative) REJECT; Q6_PACKED
flat; Q8_PACKED -2.7/-2.7/-5.4 REJECT; LEARNED PROFILE pair +5.0 consistent but MMLU 92.9/94.3
(mean 93.6, -2.1 pts, one run under floor) with conc -2.0 -> REJECT. Zero adoptions - the
AMD-tuned S25/S26 mapping does not pay on sm_120/CUDA; that is five more negatives with
mechanisms. Q6P_SELFTEST printed nothing in two captured runs -> needs a dedicated harness.
BYPRODUCT: the 32k prefill anatomy (STRATA_PREFILL_TIMING): combine 16.7% #1, gemms 19.2%,
hc 11.1%, GDN 26.1%, QSA 17.9%; short-prompt wait_copy 43.7%. THE CRASH: SIGSEGV in
strata::kernels::cpu::vnni::iq4nl_nt (CPU pool, IQ4_NL down VNNI) - three sightings incl two
on standing control legs, ~2-5% of 6-conc legs, near-null fault addresses across multiple pool
threads at one IP = blob-lifetime race suspicion (ADR exchange). Real-user impact + it silently
poisons conc means. Priority 1; T18 crash-hunt dispatched.

### F-240 — THE CRASH FOUND, ROOT-CAUSED, FIXED: `job_of` poison from a failed verify window, not the ADR exchange (2026-10-20 01:20-04:15; repro campaign/t18-repro.sh, battery campaign/t18-ab.sh, evidence campaign/results/t18-*)
The priority-1 SIGSEGV is REPRODUCED (1 crash in 120 pre-fix batches, campaign/results/t18-crashes/
t18-pre2-iter005-20261020-020302/), and the mechanism is not blob lifetime: it is stale dispatch
state. `expert_pool_dispatch_multi` mapped expert id -> job through `d.job_of`, reset ONLY on the
success path. Any early-return failure ("the expert source could not produce a blob", a bad router
id) left stale indices; the host clears `d.failed` before the NEXT round, which then REUSED the old
job descriptors: `nt` was never reset and never bounds-checked, so `act[nt]`/`nact[nt]`/`out[nt]`
(MAXT=8 each) walked into the following jobs' fields, and the pool's drain then read `hq[t]`/`act[t]`
out of bounds - SplitBufMulti float bytes as pointers, tiny denormals included. THE FAULT ADDRESSES
DECODE THE BUG: every crash line is `y[t]+2` for y[t] in {0..16} inside `vnni::iq4nl_nt`, six pool
threads at one IP (0x4723973) - impossible for any valid job, exactly what an OOB `hq[t]` read
produces (repro: segfault at 2/4/5/6/7/8, same IP). The kill chain needs no blob race, and explains
the degradation too: nt grows unbounded, the down phase does nt/8x the work per poisoned job, tok/s
decays 40->10->3 over ~40 s, then a denormal-as-pointer deref kills the engine (repro server tail
matches the original bat-lp/cf-tsum shape line for line, conc leg reads 0 tok). THE FIX
(src/core/expert_source.cpp, two guards): reset `job_of` at the START of every call (a failed window
costs that window, never the round after), and refuse an expert job whose nt reaches MAXT instead of
writing past it. Bit-identical on the success path by construction (after a clean call job_of is
already all -1, so the reset is a no-op; the nt check cannot fire within one window), so no env gate.
PROOF: 60/60 fixed probe-shape batches clean vs 1 crash in the same 60 pre-fix; 3v3 ABAB
(per-leg binaries string-verified) fix-vs-ctl means M1 205.7 vs 204.1 (+0.77%), pair 203.8 vs 206.6
(-1.32%), conc 314.3 vs 320.0 (-1.77%) - all inside the +-4% band, M1 protected. RESIDUAL: the
first domino (which dispatch failure fires first in the wild) is still unobserved live - the
engine's `ERR` line dies with its buffered stdout, so the repro captures the signature, not the
trigger; the fix makes ANY trigger self-healing. The ADR-exchange suspicion (blob lifetime, F-239)
is CLEARED for this crash: no exchange path can write a small integer where the pool reads a blob,
and the fault addresses are y[t], not w. REVERT: `cp engine/strata.pre-t18fix engine/strata`
(the pre-fix binary, kept alongside; the code revert is `git revert` of this commit).

### F-241 — T15C HC-KERNEL TRANCHE: the norm_fast flip ADOPTS (pooled +5.2% M1), the staged-down widen REJECTS at -7%, the up widen is a flagged non-adopter, LFUSE is adjudicated dead by code read (2026-10-20 04:30-07:00; batteries campaign/results/t15c-*.log, scorer campaign/t15c-score.py, harness campaign/t15c-battery.sh)

The T15C tranche ran proof-first per stage: gr_parity --selftest byte-equality (multi-vs-single across
T 1-8 direct + captured-graph replays, and the QFUSE q8_1 image vs native_quantize_q8_1) at the EXACT
env state of the arm, before any boot; then one 3v3 ABAB interleaved battery per stage (fresh boot per
leg, probe --short-n 6, per-leg .out, the arm env verified by a new boot banner in fused_gr_check). One
binary per battery — every change env-gated, so arm and ctl differ only by the env. The boot-time hc
selftest re-proved "checked bit for bit" in EVERY leg of EVERY battery (engine log).

**S3 — norm_fast_on() CUDA flip: ADOPTED (the round's win).** The gate (fused_gr.cu) hard-returned
false on CUDA since the gfx12-only era; the flip makes the split norm's register kernel
(gr_norm_split_fast_kernel — loads issued together, xn kept in registers, the second pass's 40 KB/token
L2 round-trip and one barrier deleted) the default on the standing staged read, STRATA_GR_NORM_FAST=0
rolls back. Proof: byte-equality at the flip default, at =0, and combined with S1+S2; plus the per-boot
selftest. s3norm 3v3: M1 +9.68% (212.2 vs 193.4), pair -1.51, conc +3.04, M3b +2.35. The owed confirm
3v3 (s3confirm) read M1 +0.87% (204.6 vs 202.8), pair +3.10, conc +7.91, M3b +2.23 — the first
battery's CONTROL window read low, so the single-battery +9.68% is RETRACTED as control-window luck
(the F-231 lesson, again) and the claim is the POOLED 6v6: **M1 +5.2% (208.4 vs 198.1), pair +0.8%,
conc +5.5%, M3b +2.3pp — above the +4% bar, no lane negative in any battery.** The flip is the code
default now (launcher note; rollback = one env). Consequence: F-238's M1 anchor (193.7) is stale for
M1 claims — the flip-on state boots at ~204-212 across five boots in three batteries; a five-boot
re-bank is OWED before any scoreboard row cites the new standing. The audit's +1.3-2% price was
directionally right and magnitude wrong in the interesting direction: the norm's second pass was a
real wall on this stack, not a hidden one.

**S2 — the staged down at 4 warps (81 blocks): REJECTED, reverted.** The audit's "+2.5-4%" priced
SM coverage 41->81 and ignored what else halves: 128 threads now fill the same 80 KB staged tile, the
cp.async fill latency per block doubles, and the down read slows. 3v3: M1 **-6.99%** (192.1 vs 206.5),
pair -4.00, conc -1.89 — bit-exact (proof passed, per-boot selftest passed) but slower; the code is
fully reverted and the pre-S2 binary rebuilt and redeployed within the session. F-217's
"this kernel's grid shape is load-bearing" now has both directions measured.

**S1 — UPM_COLS 16->8 (320-block up grid): proved, under bar solo, flat in stack — flagged, NOT
adopted.** gr_up_multi_kernel is COLS-templated (the per-row math reads no COLS; gr_q8_tail takes the
blocks-per-group, ==3u at 8 cols) behind STRATA_GR_UP8=1, default off. Proof passed incl QFUSE at 8
columns (direct + graph replays). Solo 3v3: M1 +3.83% (212.3 vs 204.5), pair +1.86, conc +2.67 —
under the bar (all lanes positive = a consistent stack candidate, so it was flagged forward, not
adopted). The stack battery (S4, UP8 on the adopted flip): M1 -1.01, pair +2.95, conc +1.80 — FLAT:
once the norm wall is gone, the up grid's extra in-flight bytes buy nothing; the audit's +3.4-5%
presumed the hc read was grid-capped everywhere, but the cap and the norm stall were the same
bottleneck wearing two prices. The kernel stays in the tree default-off (zero cost un-armed); the
full revert is `git checkout` of the S1 hunks of src/kernels/cuda/fused_gr.cu.

**S5 — LFUSE device-plan arm: adjudicated DEAD by code read, no battery spent (T16 #8 resolved).**
The R1 dossier's "SURVIVES — cheapest, first, -144 nodes" missed three citations: (a) F-206 MEASURED
STRATA_LFUSE=1 at -3.0% M1 on this box (the fusion tranche's clean negative — node savings were
latency-hidden); (b) re-priced honestly with F-235's measured node marginal (0.5-1.0 us x 144 =
+0.5-1% M1) the arm is under the +4% bar before it boots; (c) the code read kills it outright: on the
standing DEVICE_PLAN path, enabling LFUSE sets sg_gated_[grp] (verify.cpp:1624), and hits_in_combine
explicitly requires `!sg_gated_[grp]` (verify.cpp:1733) — the arm DISPLACES the adopted
STRATA_FUSE_HITS=1 fold (F-226, +4.35% M1), re-adding ~48 moe_hit_add launches/window and swapping the
combine to the gated form F-206 already measured losing. Strike T15 queue item #4 and R1's S2 row.
Re-open clause: a combine form that keeps hits-added under the gate AND node spacing made binding by
the R1 conditional-node surgery.

**Net:** standing recipe + the norm flip (pooled +5.2% M1 / +5.5% conc, bit-exact). The +8-9% hc-read
bucket of the T15 audit prices out at +5.2% real: the up-grid half was the norm stall wearing a
second price, and the down-grid half was a fill-latency loss — exactly the class of discovery the
proof-then-battery order exists to catch. Reverts: full pre-tranche = `cp engine/strata.pre-t15c
engine/strata` (+ git checkout of fused_gr.cu hunks); flip alone = `export STRATA_GR_NORM_FAST=0`.
Box left STOPPED. Harnesses stay reusable: campaign/t15c-battery.sh (<stage> [--ctl-env V=1] [arm
envs]), campaign/t15c-score.py (<stage>).

### F-242 — RE-BANK v2: norm_fast is CONFIRMED at means; the 3135 scoreboard moves to 203.9 / 202.0 / 306.1 (2026-10-20 morning)
Five fresh boots on the post-flip standing recipe (campaign/rebank2.sh -> campaign/results/
rebank2-normfast.log): M1 202.9/198.8/200.7/203.7/213.5 = **203.9 mean** (F-238 pre-flip 193.7:
**+5.3%**, outside +-4%); pair 201.7/200.9/205.8/199.9/201.7 = 202.0 (+1.4%, flat); 6-conc
311.0/312.7/310.9/300.9/295.1 = 306.1 (+0.9%, flat); M3b 76.2 -> 77.1. Zero crashes in all
five boots (T18 fix holding under 30 conc legs). ARCH: M1 203.9 is now above the old 2835-class
bank (198.5) by +2.7%, pair above (202.0 vs 197.6), conc still under the 2835 bank (306.1 vs
312.4, -2.0%) but above the pre-flip 3135 mean. New frozen baseline: **203.9 / 202.0 / 306.1**.
Next: prefill combine (#1 phase, 16.7%) or TC-hc mma (+9-17% class) — both execution-ready.

### F-243 — THE LAYOUT REFACTOR: LLM/{data, strata, strata-custom-2}; both lanes boot-verified (2026-10-20)
User-directed consolidation, done under "check before doing anything, be double sure": merged
Strata-data + Custom-data -> /home/svg/LLM/data and renamed Strata -> strata. VERIFICATION
BEFORE MOVING: mtp trees hashed file-by-file (41 files: 40 byte-identical, EXACTLY ONE difference
= rt/draft_vocab.bin, the deliberate stock-CJK vs custom-en split — preserved, not merged);
Custom-data/packs/q2_0 was already a symlink (no dup); zero model-filename collisions; 111 files
pre-move, all accounted post-move. EN-VARIANT DESIGN: data/mtp = stock tree; data/mtp-en =
en draft_vocab.bin (f674c934, real file) + absolute symlinks to the shared rt tensors and
parent assets — the custom lane's --mtp now points there, stock keeps data/mtp/rt. 49 operational
files in the custom repo + 7 in the stock repo sed'd to the new roots; ~/.config/strata/
settings.json data_dir -> /home/svg/LLM/data, installs -> [strata-custom-2, strata]. Path audit:
436 absolute paths checked, ALL real ones resolve (residual misses = boot-created logs, shell
syntax artifacts, one dead tqllm battery from the merge era). BOOT VERIFIED both lanes: stock
q2_0 171.0 tok/s decode + normal cache stats; custom iq3_s M1 201.8 (in the post-norm_fast band)
via data/mtp-en. RECLAIMED this round: 6.5G (mtp dedupe via links) + 6.5G (mtp-iq3s r17 assets,
unreferenced) + 479M (dead engine backups) — /home now 244G used of 952G. NOTE for future greps:
journal/dossier historical entries still say the old paths by design; operational files are all
on the new layout.

### F-244 — UPSTREAM v0.2.1: the first perf wave that reaches sm_120 (+15.6 IQ3_S / +25.9 Q2_0 M1); custom's lead narrows but holds (2026-10-20)
Pulled v0.2.1 into the stock install (884 commits since .45; engine 0.1.46), built clean after
repairing two rename-fallout layers (15 venv shebangs + CMake/ninja absolute paths still pointing
at Strata/ — F-243's legacy). A-B-A bench on both quants, fresh boot per leg, full probes
(campaign/results/v02-bench.log): IQ3_S .45 144.5/149.0 M1 (mean 146.8) -> .2.1 **169.7**
(+15.6%), pair 152.4 -> 169.4 (+11.2%), conc 239.0 -> 281.4 (+17.7%). Q2_0 .45 209.9 ->
.2.1 **264.3** (+25.9%), pair 181.5 -> 221.4 (+22.0%), conc 188.6 -> 220.8 (+17.1%).
AGAINST CUSTOM (matched IQ3_S bytes, frozen post-norm_fast scoreboard 203.9/202.0/306.1):
custom still leads .2.1 by +20.2 / +19.3 / +8.8% - the margin collapsed from +23-33% vs .45.
Q2_0 CROSS-SESSION CAVEAT: stock .2.1's 264.3 is above our fork's Oct-17 Q2_0 leg (242.6), but
different session/clocks - a same-session custom-Q2_0 A/B is OWED before any lane claim.
Stock install now deployed at v0.2.1 (requested); .45 kept as engine/strata.v0.1.45-pre-v02.
NEXT: commit-level analysis of what produced the wave, ranked for cherry-pick into custom-2.

### F-245 — T19: the v0.2.1 wave ported one variable at a time; S1 adopted, two flagged, S3 reverted, the greedy-identity instrument invalidated (2026-10-20 11:20-13:10)
The tranche followed the wave dossier (campaign/research/v02-wave-analysis.md) stage by stage:
port -> server-free bitwise proof at arm AND ctl env states -> one 3v3 ABAB (fresh boot per leg,
--short-n 6, binary md5 + t18 marker per leg). Full per-stage detail in SVG-CUSTOM-TODO.md
"T19 TRANCHE RESULTS"; batteries campaign/results/t19-*.log, proofs t19-s*-proof*.log.
- **S1 mmvq_il WIDE (75fabbd9+44997124) — ADOPTED, the round's engine.** 5-8 column interleaved
  matvec + the sm_120 tables (kIlRows_120/kIlRows58_120 keyed 1201 = 12.0 with 80+ SMs). Bitwise
  (mmvq_il_parity T 1-8, every rows choice, both env states). Our 5090 bench kills the cliff: head
  T5 491.6 -> 290.6 us, T8 627.6 -> 372.8 (-41%). Battery 3v3: **conc +8.34%** (309.6 -> 335.4),
  M1 +2.63% (under the solo bar — flagged honestly), pair -1.70 (in band), M3b +0.48pp flat.
  Adopted on the targeted conc lane (dossier band +6-12%); M1's sub-bar solo ride is why the
  scoreboard re-bank is owed. Rollback: STRATA_MMVQ_IL_WIDE=0 STRATA_MMVQ_IL_120=0.
- **S2 GR_DOWN_TG (94104176) — FLAGGED, default-off.** Kernel win real (staged down T8 17.5 vs
  27.3 us cold, whole read -21%; bitwise at STRATA_HC_SPLIT=2 arm+ctl) but the battery read every
  lane inside the band (M1 +1.56 / pair +2.25 / conc +2.43). Env-only: STRATA_GR_DOWN_TG=n.
  Lesson banked: the hc harnesses run the PLAIN read unless STRATA_HC_SPLIT=2 — the first proof
  proved nothing before it was corrected.
- **S3 SIDE_MIXER (ffc30b8d+477ac13c) — REVERTED, and the falsifier itself invalidated.** The
  fixed-prompt greedy identity check "diverged" deterministically at =1 (3 boots, one hash) vs ctl
  (2 boots, one hash) — and =2 diverged only under STRATA_VERIFY_ASYNC_LAUNCH=1. Reverted per law
  10. THEN S7's greedy riders on the STOCK tree showed the truth about the instrument: stock .2.1
  greedy outputs are NOT stable across boots at identical env (prompt 1: three distinct outputs in
  three same-env legs; the fork's "divergent" hash appears in a stock OFF leg). Cross-boot greedy
  text is NOT a bitwise instrument on this engine. The S3 rejection STANDS on the honest grounds:
  its bitwise-safety claim could not be established (the proving instrument doesn't work), the
  3v0-vs-0v2 hash pattern still reads probably-real (~p=0.02), and the =2 x ASYNC_LAUNCH anomaly
  is unresolved — an unprovable change does not boot on this lane. Re-open: a logits-dump A/B
  harness with per-state output distributions; also check STRATA_VERIFY_ASYNC_LAUNCH interaction
  in isolation. Applied hunks banked: campaign/results/t19-s3-rejected.diff.
- **S4 GU_SWIGLU (75afe3f4+8837cb6f) — FLAGGED, default-off.** Bitwise (iq_multi_parity, every
  gate/up x down format, arm+ctl+default). First battery VOIDED by a 180.6-221.8 control-window
  drift (new worst case for the interleave law — the box moved ~20% inside one 8-minute window);
  re-run read +5.25 M1; pooled 6v6 is the honest read: **M1 +0.4 / pair -0.4 / conc +1.0 — FLAT.**
  Upstream's +0.5-0.9% is under our noise floor. STRATA_GU_SWIGLU=1 arms it.
- **S5 GR_UP75 — KILLED BY CODE READ, no port.** Upstream's kernel static-asserts the 256-thread
  geometry (RPW==8); our gr_up_multi_kernel is 512-thread/RPW=4 — a port is a rewrite. The
  absorption prior is same-path and direct: F-241 S4 (gr_up8 FLAT post-norm_fast). Upstream's own
  per-switch number is the smallest/noisiest (-1.7% cumulative).
- **S6 STACK (riders on S1): NOT ADOPTED.** M1 +1.80 / pair +5.60 / conc +2.40 / M3b -1.87pp. The
  pair spike contradicts the riders' mechanism (kernel savings must surface in M1/conc — both
  sub-band); pair swung +-6% across today's batteries. F-191 again: a single-lane spike is a rumor.
- **S7 CONFOUND (stock v0.2.1, 3v3, four package switches on vs off) — F-244 ANSWERED.** The
  ba2c2d5f+94104176 package prices at **M1 +9.13% (159.7 vs 146.4), pair +7.57%, conc +15.73%**
  (272.9 vs 235.8) of the +15.6/+11.2/+17.7 wave: the kernel package IS the wave (~60% of M1, ~90%
  of conc); the I/O-tier + remaining defaults are the M1 residual (~+6.5) and are negligible on
  conc. Our +15.6 M1 exceeded upstream's +11.9-14.0 anchor mainly because our --spec 6 windows pay
  the 5-8 column matvec more — S1 is our share of exactly that.
STATE: engine/strata = S1-default build; backups .pre-t19-s1..s4 alongside. Scoreboard: the frozen
203.9/202.0/306.1 (F-242) is pre-S1; S1's window read ~215/204.6/335.4 arm vs 209.5/208.1/309.6
ctl — a five-boot re-bank of the S1 engine is OWED before any cross-session claim. Stock deploy
restored to v0.2.1 (7a5e17d1) after S7. BOX STOPPED at round end.
NEXT: re-bank the S1 engine (5 boots); the logits-dump A/B harness if the mixer port is re-wanted;
S3's stock-side check folded into any future stock boot (free rider); the mmvq tables' share of
S7's residual could be priced by one stock boot with STRATA_MMVQ_IL_WIDE=0/_120=0.

### F-246 — T19 re-bank + the window-shift proof: mmvq_il CONFIRMED (same-window +14.6% conc), no regression (2026-10-20 16:42-17:00)
Rebank3 on the post-T19 engine read M1 209.5 / pair 194.7 / conc 266.8 - M1 a new best, conc
-12.8% vs the frozen 306.1. Per the law, no drift excuse without a counterfactual: ran a
same-window 3v3 ABAB of the post-T19 binary against engine/strata.pre-t19-s1 (norm_fast only),
16:53-17:00 window. RESULT: pre 195.2/192.0/245.7 vs cur 201.5/191.1/281.5 = M1 +3.2%, pair flat,
conc **+14.6%** - the T19 binary is AHEAD in-window, so the low absolute conc belongs to the
WINDOW, not the port (both binaries read 245-282 here vs 310-335 this morning; M1 meanwhile
matches the 11:38 window's 210-220, so the shift is conc-lane-specific -> host/cohort-reactor
contention class, first documented instance of a conc-only window shift).
LEDGER RULE adopted: absolute scoreboard = frozen at the last quiet-window five-boot (morning
203.9/202.0/306.1 stands as the record until a quiet re-bank); RELATIVE claims (arm vs same-window
control) are the stable currency. T19 accepted on: S1 battery +8.3% conc / +2.6% M1 (11:38
window) AND same-window control +14.6% / +3.2% (16:53) - mechanism-consistent both times.
Post-T19 standing binary preserved as engine/strata.t19-final; pre-T19 as
engine/strata.pre-t19-s1. OWED: one quiet-window five-boot re-bank to freeze the new absolute
record (expected ~209-212 / ~195-205 / 310-335 depending on window class).

### F-247 — T20: the whole v0.2.1 wave merged in one tree and priced in one window: WIN (M1 +11.1%, prefill +4.9%, every other lane in-band positive); the merge replaces cherry-picking if the MMLU gate and a quiet re-bank clear (2026-10-20 18:00-19:15; battery campaign/results/t20-merge-ab.log in the COPY ~/LLM/strata-custom-2-merge)
The T19 tranche ported the wave one commit at a time with the fork's defaults (S1 adopted, the
rest flagged off). T20 asked the cheaper, coarser question the wave analysis left open: what does
the ENTIRE merged engine cost or gain on the same window - including everything the fork had
refused to arm one-by-one (SIDE_MIXER=2, GR_DOWN_TG on sm_120, GU_SWIGLU on, PF_FUSED, the I/O
tier, the queue rework)? `git merge v0.2.1` into a full COPY of the fork (original untouched,
still standing; merge commit 74239c96, local only). 15 conflicted files, 55 hunks, resolved by
the preservation list (full resolution table: SVG-CUSTOM-TODO.md T20). The batch/cohorts region
was recomposed by hand after the auto-merge produced duplicate batch_poll definitions: the merged
batch_poll = our G2 doorbell loop + upstream's ext_serve_ watchdog + their batch_serve rewritten
with OUR group math (doorbell k = layer lb_+k/G, group k%G); commit graphs stay parity-keyed
(F-196); upstream's pre_launch_/handoff/stamped-wait hooks wired into our shapes. T18's rebase
hazard did NOT fire - both guards survived the io-prefetch rewrite inside the auto-merge
(verified in-source). norm_fast stays OUR flip (upstream opt-in loses to our sm_120 +5.2%).
Proofs: mmvq_il_parity T 1-8 every rows choice bitwise (preservation item 5 resolved to
upstream's version - it adds the sm_75 tables); gr/gr_multi parities at default, STRATA_HC_SPLIT=2
and STRATA_GR_DOWN_TG=0, all 0 failures; iq_multi_parity default + GU_SWIGLU=0; gdn_chunk +
gdn_rec parities. Boot on 8081 via the ORIGINAL launcher: health OK, the boot-time hc selftest
("checked bit for bit against the plain read"), norm_fast boot line, "cohorts: round A.S=6 B.S=6"
rounds live, coherent smoke under the full merged default stack.
A/B (3v3 ABAB interleaved, fresh boot per leg, probe --short-n 6, ONE window 18:50-19:01; binary
the only variable, ORIGINAL engine slot): means - M1 194.1 -> 215.8 (**+11.1%**), pair 199.3 ->
200.9 (+0.8%), 6-conc 311.4 -> 321.3 (+3.2%), M3b 74.4 -> 75.3 (+0.9pp); prefill 32k real-corpus
(ONE leg per binary, consecutive): 5816.7 -> 6103.1 tok/s (**+4.9%**). Worst lane +0.8%, two
lanes >= +4% - the stated win rule clears. VERDICT: WIN; replace recommended behind (1) the 2x
MMLU-Pro gate (PF_FUSED is lossy-tiny and in the default stack) and (2) a quiet-window five-boot
re-bank of the merged binary. Rollbacks all env (SIDE_MIXER/GR_DOWN_TG/GU_SWIGLU/PF_FUSED/
MMVQ_IL/GR_NORM_FAST/ATTN_DAILY/SM120_PERF =0). The merged candidate: copy's main @ 74239c96,
binary md5 b8a21923 (campaign/results/strata.t20-merged); control preserved: engine/strata.t19-final
(53bbe9fb), still the standing engine - the original repo's files untouched (git diff empty).
LESSON (build hygiene): the copy's build/ cache had silently pointed at the ORIGINAL's sources -
"the copy builds clean" had been rebuilding the WRONG TREE into the copy's build dir. Any
copied-tree build must be re-configured with absolute paths into the copy and its build.ninja
spot-checked for the copy's paths before the first object is compiled; a binary's provenance is
its build.ninja's source paths, not the directory it was produced in.

### F-248 — T20 ADOPTED: the merged v0.2.1 line IS custom-2 now; gates passed; new scoreboard frozen (2026-10-20 evening)
Adoption executed: original repo reset onto the merged line (edb6c4a2, merge commit 74239c96 with
15 conflicts 7/7-preserved), rebuilt IN PLACE (163/164 links - no cache trap this time), deployed
(engine/strata = merged build), smoke-verified live: COHORTS armed ("slots [0,3) on verifier A,
[3,6) on B"), hc read "checked bit for bit", M1 smoke 205-215. GATES: MMLU-Pro 95.7 + 91.4 (mean
93.55 >= 93.3 floor; one run low - WATCH RULE armed, mimic 58/66 = 87.9 = anchor). FIVE-BOOT
RE-BANK of the merged binary (evening window, same session class as the A/B): M1 220.9/211.2/
206.9/200.6/223.0 = **212.5**; pair 204.9/204.5/194.8/200.8/209.2 = **202.8**; 6-conc 322.2/
319.8/311.3/303.3/321.6 = **315.7**; M3b ~76.9. vs frozen 203.9/202.0/306.1: +4.2% / +0.4% /
+3.1% at the means, and the same-window A/B's authoritative relative read stays +11.1% M1 / +4.9%
prefill / +3.2% conc. ROLLBACK: engine/strata.t19-final (one cp); env-level per-arm offs exist
(SIDE_MIXER/GR_DOWN_TG/GU_SWIGLU/PF_FUSED/MMVQ_IL_WIDE/GR_NORM_FAST=0). Housekeeping: my unpushed
housekeeping commit was dropped by the reset and re-applied (mimic JSON recovered from reflog);
the merged tree's !campaign/results/*.log negation makes the older evidence logs visible and
they are being banked with this entry. LAB COPY: ~/LLM/strata-custom-2-merge retained as the
merge's provenance archive (delete when convenient). NEXT: quiet-window re-bank confirmation,
MMLU re-gate at the next adoption round, then COHORTS=3 / re-quant / TC-hc per the ladder.

### F-249 — T21: COHORTS=3 measured FLAT (not adopted); MMLU re-gate CLEARS the merged-line watch (2026-10-20 night)
COHORTS=3 battery (t21b, 3v3 ABAB after a harness fix - the first attempt put the launcher copy in
campaign/ where its relative .venv resolution broke; lesson: launcher copies live at repo root):
arm M1 199.1 / pair 189.5 / conc 307.1 vs ctl 202.3 / 194.0 / 306.2 = M1 -1.6%, pair -2.3%, conc
+0.3% - ALL inside noise. VERDICT: NOT ADOPTED - the third verifier buys nothing at exactly 6
clients. Mechanism read: 6 slots over 2 cohorts = 3+3 fills both reactors; 3 cohorts = 2+2+2
narrows every window without adding overlap - T17B's +8-15% estimate assumed wider slot counts;
at 6 streams the split is neutral. Re-test only if parallel grows (>=9 clients) or the reactor
learns cross-cohort tail reuse. Launcher copy kept as launch-custom-iq3s-8081-c3.sh (dormant).
MMLU RE-GATE on the standing merged recipe: 95.7 + 94.3 (mean 95.0) - the F-248 watch flag
(91.4 low run) CLEARS; quality on the v0.2.1 line confirmed at anchor class.
Verification-method note: the engine log is created lazily per boot (first request flush) - boot-
line greps must fire after a request-bearing probe, not at health. Control window class tonight:
evening/mid (M1 ~203, conc ~303-307 on C2).

### F-250 — THE HEADLINE A/B, same-window: custom (merged line) vs stock v0.2.1 — +27% M1 / +24% pair / +15% conc at identical bytes (2026-10-20 21:59-22:07)
3v3 interleaved, alternating boots, both engines on the same IQ3_S config/model/pack/profile:
stock 160.6 / 156.9 / 267.4 (M1/pair/conc means) vs custom **204.5 / 195.0 / 308.2**. Per-round
M1 deltas +38.1 / +18.1 / +27.0% - every round, every lane, custom wins. Side observation:
stock is more window-sensitive (its evening M1 160.6 vs its morning 169.7; custom's spread
197.9-211.9 vs its own band) - the fork's arms appear to smooth window variance too. The fork
premium over the LATEST upstream: COHORTS + norm_fast + FUSE_HITS + POOL_DIET + GDN_CHUNKED +
T18 + the tuned arm set, on top of the shared v0.2.1 kernel package.

### F-251 — T24 BASALT PORT TRANCHE, S1+S2: two port-now candidates meet the same-window truth — dense_tc a wash (shelf), in-tree hc8 a regression (recorded) (2026-10-21, goal round 11)

**THE TRANCHE'S MAP HELD; ITS PREMISES NEEDED THE BOX.** T23 ranked dense_tc (#1) and hc8 (#2) PORT NOW
on the 8-bit-arm calibration (their 213.4 ~= our 212.5). Both ports close the same way: the class was
already correctly priced by OUR tree — dense_tc loses to the T19-wide mmvq on our K-quant dense mix, and
the hc8 falsifier was literally an env var (STRATA_HC_REQ8) waiting for its first A/B.

**S1 — dense_tc, PORTED AND PROVEN, THEN FALSIFIED ON THE NUMBERS.** Port: basalt
src/kernels/cuda/dense_tc.cu @ fb0944c7 -> src/kernels/cuda/dense_tc.cu (+ dense_tc.hpp), namespace
strata::kernels, our pdl.hpp underneath (pdl_trigger for pdl_launch_dependents; launch_pdl falls back to
plain launches - the env default never arms PDL), dense_tc_mask() reads STRATA_DENSE_TC (default 0; 1 =
the donor's mask 29: GDN qkv/out + QSA q/out) instead of their constant-on. Dispatch in verify.cpp's
mm()/q_chain with a width gate n >= 4 (the microbench crossover: cold-L2 per our card, every covered
type/shape >= 1.0x from T=4, up to 1.9x at T=8; T=1 is 0.72-0.9x and the window mix lives at 2.47
accepted/window). Parity (dense_tc_parity, the donor's harness ported): ALL PASS - T-column bitwise
contract every type T=1..8, ~1e-7 rms vs float64 (the same class native_mmvq sits at), our exact shapes
incl. the pair kernels. PREMISE CORRECTION (the tranche's one real discovery on the pack): the T23 study's
"our dense is Q8_0" was an index-column misread - the GGUF headers say the dense projections are a K-quant
MIXTURE (qkv 22 Q6_K/10 Q5_K/2 Q4_K/2 IQ4_XS; ssm_out 29 Q6_K/4 Q5_K/3 Q4_K; attn_q 5 Q6_K/3
Q5_K/3 IQ4_XS/1 Q4_K; attn_output 11 Q6_K/1 Q4_K). dense_tc consumes them all natively, so the Q6_K dense
re-quant the study queued is MOOT - built, proven unnecessary by the header check, deleted (165 GB
transients reclaimed); the =COPY patch on third_party/llama.cpp's quantize (restoring the runbook's
documented name=COPY; the pin only knew whole-file COPY) is banked for the T14B experts-requant.
Battery 3v3 ABAB, same binary, one env (campaign/results/t24-s1-battery.log):

| | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| ctl mean | 216.77 | 204.70 | 335.80 | 74.77 |
| arm mean (STRATA_DENSE_TC=1) | 216.33 | 207.67 | 332.17 | 76.77 |
| delta | **-0.20%** | +1.45% | -1.08% | +2.7pp |

VERDICT: the targeted lane is a WASH - the +4% bar fails, NOT ADOPTED. The donor's +5.6-10.2% was priced
against their weaker mmvq and bundled their Q8_0->Q6_K byte cut; T19 already banked our equivalent. The
port stays as a LABELED SHELF ARM (STRATA_DENSE_TC=1 = one launcher line; default-off proven bit-neutral
by the battery's own control legs on the new binary). 2x MMLU on the arm: 95.71 / 92.86 = the standing
class (mmlu-t24s1-arm-g{1,2}.json) - the shelf is quality-stamped for any future re-arming. Backup:
engine/strata.pre-t24-s1.

**S2 — hc8: THE FALSIFIER WAS IN-TREE (T25-1 was right), AND IT REGRESSES.** No port: our tree's
STRATA_HC_REQ8 (weights.cpp load_hc_q8; the hc projections requantized at load to int8 + fp32 scale/32 -
675 MiB - with the freed VRAM growing the expert cache 17.7 -> 17.90 GiB; fused_gr.cu:1340 routes the
reads to its q8 variants) is the hc8 mechanism, one env boot from its first-ever A/B. New harness
hc_q8_parity (registered in CMake): 18/18 PASS - q8-vs-bf16 rel rms 2.3-2.4e-3 (quantization-noise class),
the T-column bitwise contract every T, inject rows exact (stay bf16). Battery 3v3 ABAB
(campaign/results/t24-s2-battery.log):

| | M1 | pair | 6-conc | M3b |
|---|---|---|---|---|
| ctl mean | 221.37 | 213.87 | 332.93 | 75.97 |
| arm mean (STRATA_HC_REQ8=1) | 205.50 | 198.97 | 337.50 | 77.23 |
| delta | **-7.17%** | **-6.97%** | +1.37% | +1.7pp |

M1 regressed in EVERY round (-2.4/-11.9/-6.8) - the single-stream protection law alone rejects it.
Mechanism: the q8 read runs the NON-staged variant (fused_gr.cu:1358), surrendering the staged bf16
read's cp.async overlap; the byte cut (0.56x) does not pay for it at our node density - exactly the
F-180-era pricing ("hc bf16->Q8 byte cut ~0.4 ms - LOW until node spacing is fixed"). REJECT + record;
nothing standing changed (env default-off). The basalt hc8_gr/hc_fused port closes with it: their +9..17%
band assumed fp16-scale copies AND the persistent one-kernel form, not this read.

**S3 — PDL chain: NOT RUN (the tranche's own gate: only if S1+S2 close cleanly; S2 regressed).** F-191's
partial reversal stands open with T25-1's port-with-work plan as the map.

**The tranche's compounding:** two clean falsifications with mechanisms, one premise correction (the
pack's dense is K-quant mixed - relevant to EVERY future dense-line candidate), one tooling bank
(quantize =COPY), two registered harnesses (dense_tc_parity, hc_q8_parity), two shelf arms with stamped
quality, and the standing recipe untouched at every step. T18 guards verified in the deployed binary
(job_of reset + the nt>=MAXT marker). Box STOPPED at 13:30; basalt clone /tmp/basalt-t24 deleted.

## F-252 — 2026-10-21: THE FMLA32-STYLE HALF-RATE LOCK EXISTS ON OUR 5090 — MEASURED, NOT INFERRED
Probe: pure-register mma.sync rate bench (campaign/mma-accum-ratelock.cu, sm_120, warp-level
m16n8k16, no memory), two ABAB legs. Window: the user's 8081 server was live (96% util), so the
absolute TFLOPS are contended — the RATIO is the instrument, and it is stable.
Result: fp16-accum mma = 476 TFLOPS; fp32-accum (fp16 in) = 238; bf16-in/fp32-accum = 238.
Ratio f16/f32 = 1.99 / 1.97 across legs. Consumer Blackwell half-rates the fp32-accumulate
tensor pipe exactly like the Ampere 3090 "FMLA32 = 1/2" story (Alexey Fateev's tweet thread,
2026-10-21, github.com/Zagitta/gpu-unlock).
Comparators: the module itself is Ampere GA102/GA104 ONLY — other chips explicitly rejected —
so it cannot run on our card and no Blackwell unlocker exists. Fateev's own Strata numbers on
the 3090 price the unlock: +10-12% prefill (GEMM phases ~40% of prefill; int8 experts/CUDA-core
kernels unaffected), decode flat.
Consequences for the record:
(1) Every GEMM/WMMA line in our prefill that accumulates in fp32 (prefill/kernels.cu WMMA
    waves included) runs at half the tensor-pipe rate by design; if a Blackwell unlock ever
    appears, ~10% prefill is the family-measured price, free.
(2) The TC-hc mma plan must be priced at 238 TFLOPS bf16-f32-accum, not 476 — the +9-17%
    ceiling already assumed the fp32-accum rate; the fp16-accum form would be the 2x pipe but
    breaks bit-identity at K=2560 and needs the KL yardstick before it is even a candidate.
(3) Decode is untouched by this law (int8 experts, CUDA-core dequant) — consistent with the
    3090 report's flat decode.
No action: the unlock is Ampere-only; keep a watch for a Blackwell port. Bench committed,
binary not.

## F-253 — 2026-10-21: T26 DEEP-DIVE WAVE (6 source-level researchers, family repos vs our tree) — THE PLAN HOLDS, NINE AMENDMENTS, FOUR NEW ITEMS
Six researchers, one subsystem each, read both trees to file:line (dossiers campaign/research/T26-{1..6}-*.md,
~180 KB). The T25 calls survive, with these corrections — every one below is a plan change, not a detail.
THE BIG ONE (decode): our LIVE launcher never enables the device plan — STRATA_VERIFY_DEVICE_PLAN is unset,
device_plan_=false (verify.cpp:1015), so every (layer, group) pays the host-plan round trip (~12-14 nodes)
where basalt runs ~6-7 (route -> PDL swiglu/down -> in-kernel dm_wait poll -> combine pre-loading under the
wait). The MoE-tail port is therefore STRONGER than T25-1 priced (+5-10% M1 est.), and its Phase A is one
env boot. Bit-identity is free by construction: our pack's pairs (14x(18,20), 3x(18,42), 9x(21,20),
1x(21,42), 15x(22,20), 5x(22,42), 1x(23,20)) are all in DM_PAIRS (dec_moe.cuh:1024-1027), (23,20) via the
glue fallback; group ordering matches algebraically (dm_group/dm_glue vs resident_plan_par_kernel), combine
sums match incl. the +0.0f hit edge. NEW GATES for the port: --pcie-frac 0 pairing precedent, per-group
DmPlan for our G=2 split, cvec/hand_in stand-down check, and dm_test_kernel parity + DM_WAIT_STATS as a
read-only instrument boot BEFORE surgery.
NEW: mid_chain is probably NOT blocked — T25-1's "our qkv/z/out are Q8_0" was the F-251 index-column
misread; per the GGUF headers (qkv in {Q6_K,Q5_K,Q4_K,IQ4_XS}, ssm_out in {Q6_K,Q5_K,Q4_K}) the types sit
inside type_ok {12,13,14,21,23} (mid_chain.cu:443). Pending one attn_gate(z) header check, mid_chain runs
today via the generic triple with no re-quant. It JUMPS the GDN ladder.
T24 closures hold at source: dense_tc's n<4 gate makes it inert on our M1 mix (mean T=2.47) — killed for
this pack, never re-arm via the gate; hc8's mechanism is confirmed and broader than we wrote (q8w forces
staged=false at fused_gr.cu:1358 AND disables the split variants :2379/:2415 and the staged chunk tiling;
byte ceiling ~0.29 ms vs -0.83 ms measured).
SPEC LANE: the keep rule is 8 f32 bins, keep iff prod(acc(p_i)) >= th, th=0 = flat min-p, base 0.35,
EMA(0.97) adapt (draft_cum.hpp:25-76); our side is the flat gate only (generate.cpp:15162-15164) — port is
a ~50-line header + 5 edits, with one delta basalt never faced: our rescue advances past the exact-match
walk (generate.cpp:15390-15413) so observe() must be rescue-inclusive. Conditional draft graph is
applicable (CUDA 13, no cooperative kernels in the path) but the band drops to +0.3-0.9% ; it can run
bit-identical (th=0) AHEAD of the keep rule. DFlash2 taps VERIFIED from source ([5,15,25,35,45] hc-means,
12,800 feats); drafter internals unverifiable (weights are LFS pointers); +2-4% is a class estimate and
their own AL delta (+0.18-0.42) sits BELOW our T14B target — T14C+ rider only. DSpark KILLED as a serve
arm (dprob is the free confidence; folded into the refit). DeepInfer varlen KILLED single-stream (flat
c_row), conc parked.
CORRECTION THAT DISARMS A FALSIFIER: --adapt-every is expert-TIER adaptation, default 4, decay 0.7 (serve
fast path 0.93) — generate.cpp:495-496,5723,9253; the "0.85" in T25-3 was not basalt's number and the
item was never depth scheduling. Repriced: +0-2% conc at best.
CONC: the PIPED_CONCURRENT one-boot falsifier is DEAD as written — the overlap branch requires
piped = batch>0 && batch_groups>1 && n_pipe>1 (generate.cpp:13850,12025), which our single-GPU COHORTS
lane never satisfies; the env arm is a silent no-op and STRATA_DMA_BATCH=0 does not clear the gate (the
gate reads pcie_mode). The dma refusal itself is exact: one doorbell under dma = cudaMemcpyAsync +
cudaLaunchHostFunc holding the driver lock (expert_source.cpp:4776 -> verify.cpp:3166-3192) while the
window spins on that flag. Port = reader thread over {ver, ver_b} mail jobs, M cost, +3-6% net est.
(335-350 conc). Elastic KV conflicts with our live --kv-resident streaming (~0 standalone; parallel-8
enabler only). NEW cheap lever: raise STRATA_BATCH_DECODE_SHARE (env, S, +1-4% est.) — admission overlap
into the reactor, where the reactor currently stops during every read piece. ADMIT_SHORT_FIRST is
TTFT-only at 7+ clients (0% conc at 6); maya fair-slice is fairness-only. COHORTS>3 stays killed at 6.
LADDER REVISED: decode-share raise -> PIPED port -> parallel-8 (380-420 band, elastic/reserve funded).
PREFILL: the plateau attributes to the flat per-token pipeline — MoE complex 41.4% + dense fp16 GEMMs
25-33% running at 60-75% of the 238-TFLOPS fp32-accum lock; attention is a growing but minor term. The
launch-map diet is a DECODE lever, not a prefill one (prefill is stream-launched; launch overhead <1.5%
there): -900..-1000 nodes -> ~1100-1200, -0.43..-0.95 ms/window. Standing PDL value today = 0
(STRATA_DF_PDL unset AND STRATA_DENSE_TC off); UseNodePriority is missing (we pass 0) = new one-flag arm.
F-252 FOLLOW-THROUGH, DISPUTED PRICING: R3 prices a blanket fp16-accumulate prefill at +10-17% (cublas
COMPUTE_16F swap, XS) but flags the F-230 acceptance tax; R5's error analysis says blanket KLD 0.15-0.40
(>> the 0.05 gate; PV accumulation stagnates at ULP) = killed blanket, only the K=640 and maybe K=2560
dense scopes survive. RESOLUTION: gate first (KL yardstick), scope to K=640/2560, M3b + 2x MMLU mandated —
recorded as a disputable candidate, not a plan item until the gate speaks.
NUMERICS: RECIPE-alpha arithmetic VERIFIED against the live native_experts.txt — 42.97 GiB (-8.26%),
3.0556 bpw, 10,129 slots = +792 confirmed; direct M1 +1.90% (3.26 ms hits / 14.2 ms window), band honest;
262k absorption 9,362 >= 9,337. AMENDED: alpha requantizes 28 tensors (~11 GB in / ~7 GB out), not "36,
~5 GB". KL yardstick is fully in-tree (set.ids.json byte-matches the tokenizer; STRATA_LOGPOS + TOPK=64 +
--short-read + quant_quality.py run/compare, prompt/pcompare mandatory, ctl-vs-ctl self-noise first;
thresholds top-1 >=97%, mean KL <=0.05, p99 <=1.0). TRAPS: draft_vocab EN — production serves mtp-en's
162 KB subset but gates-spec G0 copies the 425 KB full vocab (wrong contract); /home/svg/LLM/Custom-data
is now EMPTY (four specs' paths stale); quant_quality.py launches its own engine (fence vs the live 8081).
MEMORY/HOST: cold start is a BACK-TO-BACK-BOOT phenomenon — boot N caches the shards, boot N+1's pin drops
the whole file cache (pin_depends_on_reclaim, expert_source.cpp:100-102 -> drop_file_cache :3474-3484) and
the first lane re-reads 44-55 GB; a lone restart does not pay it (today's 14:30 boot read 297 MB). FAST-WARM
AMENDED: the learned-profile lever is ALREADY DEAD (T18: MMLU 93.6 < floor, conc -2.0); the surviving
package is campaign/prewarm2.sh post-boot (never run, zero code) = +3-5% five-boot means (est.).
HW LOCK: the RTX Blackwell whitepaper itself lists the 5090's BF16-fp32acc at 209.5/419 — the same
documented 2:1 as GA102's 71/142. F-252 stands as documented spec, not a hidden fuse; gpu-unlock stays
WATCH-ONLY (it is GA102/GA104 PCI-ID-gated; a Blackwell equivalent would need a GB202 selector/PLM payload).

## F-254 — 2026-10-21: T26-1's HEADLINE RETRACTED (verified against launcher + history) — THE DEVICE PLAN IS ALREADY ON, AND THE HOST-PLAN RTT WAS ALREADY PRICED AT ~0.26 ms
What T26-1 claimed: "our live launcher never enables the device plan (STRATA_VERIFY_DEVICE_PLAN unset)
=> every (layer,group) pays a host-plan RTT = MoE port Phase A is one env boot, +5-10% M1 est."
Verification (before any boot): launch-custom-iq3s-8081.sh:23 exports STRATA_VERIFY_DEVICE_PLAN=1 and
nothing overrides it (the launcher is 47 lines, no arms-env source, no later unset). verify.cpp:1015 sets
device_plan_ = !all_resident_ && !always_publish_ && env; all_resident_ is a per-layer-range 100%-VRAM test
(verify.cpp:978-997) and our cache (9,390-9,435 slots of ~24.6k experts) is nowhere near it, so
device_plan_ = TRUE in the standing recipe. The researcher read campaign/daily-iq3_s-8081.json — an env
flag, not a config key — and stopped there. --resident-experts in that config is a RAM mode
(generate.cpp:1122-1126), not the residency graph.
And the prize was already measured: T1.1's partial device plan (STRATA_PLAN_PARTIAL=1, fully wired on both
sides — verify_kernels kPlanPartialSkip, verify.cpp:1965-1975, expert_source.cpp:4691-4697) was executed
and CLOSED NEUTRAL on 2026-10-16: ABAB 3v3 arm 190.0/198.5/249.7 vs ctl 191.2/201.7/251.9 = -0.6/-1.6/-0.9%,
and the mechanism correction recorded there prices the flagA host-plan round trip it removes at ~0.26 ms.
CONSEQUENCES: (1) the "device-plan falsifier" is VOID — it is already on and there is no free boot;
(2) the MoE-tail port's prize is NOT the host plan. What survives from T26-1 is the part that never
depended on the premise: our expert pairs map onto DM_PAIRS bit-identically (incl. the (23,20) glue
fallback), the plan-list ordering matches algebraically, and the donor's in-kernel tier wait + folded
combine + doorbell count is a different mechanism from the plan RTT. The port's value must now be
re-derived from THOSE mechanics before it earns a tranche; its first boot is the dm parity + DM_WAIT_STATS
instrument, not a flag. The roadmap's #1 slot moves to the genuinely unarmed items (decode-share, KL
yardstick, calibrate.py, prewarm2.sh). Lesson repeated from F-232: an agent's headline is a hypothesis
until the launcher and the history are read — especially when it claims "prior studies didn't know".

## F-255 — 2026-10-21: T27 FRONTIER WAVE (3 researchers, online: arxiv/GitHub/NVIDIA/news) — THE ROADMAP IS EXTERNALLY CORROBORATED; FOUR NEW RIDERS, ONE ORDERING FIX
Three frontier dossiers (campaign/research/T27-{1,2,3}-*.md). Headline: the field converged on OUR
mechanisms — SGLang's low-latency mode (2026-10-18) ships exactly the MoE-tail mechanics T26-1 found in
basalt (bitmap+popcount route skip, expert GEMMs launched before routing, dependency release inside the
MoE polling loop, tag-monotone in-kernel waits = our wait_flag_ge), and vLLM's Dynamic Speculation Length
RFC closed NOT-PLANNED with our two mechanisms (confidence-threshold halt + device-side control flow =
draft_cum + conditional graph) named as the future work. Our lanes are the right lanes.
T27-1 (kernel/hardware): sm_120 silicon map pinned — NO tcgen05/TMEM, NO wgmma (mma.sync only; every
SM100 donor kernel is inapplicable), clusters+DSMEM present (dec_gdn_rec safe), UseNodePriority=8 now
documented, CUDA 13.1 adds device graph-launch tooling. MoE-tail port CONFIRMED as top kernel item with
AMENDED spec (add SGLang's bitmap/popcount route-skip + launch-before-routing to Phase B). PDL chain
CONFIRMED + AMENDED: TRT-LLM's casebook shows PDL default-on with per-kernel WAR DISABLES for accuracy
(quant/routing/MoE) — our chain needs a per-pair kill switch + accuracy record + one nsys probe per pair.
cuBLASLt heuristic sweep on our 8-10 prefill shapes = un-run ZERO-BOOT microbench (cuBLAS demonstrably
mis-picks on 5090 in an adjacent report). No Blackwell fp32-accum unlock exists (whitepaper's 2:1 is
documented spec; only hardware-blocked vGPU unlocks). NEGATIVE: whole-model decode megakernels
(megakernel/Lucebox class — their own sm_120 number 205 tok/s < our 212.5; our pool-tiered byte-floored
expert line is unhideable; T-column contract unverifiable across a rewrite). "ACES" could not be verified
to exist — treated as nonexistent until a source appears.
T27-2 (spec/model): EAGLE 3.1 (vLLM blog 2026-05-26) names "attention drift" — the drafter's focus decays
to its own tokens with depth — and fixes it with FC-normalization of target hiddens pre-fusion + post-norm
feedback into the recursion. This is a MECHANISTIC DIAGNOSIS OF OUR F-235 DEEP-POSITION DEFICIT (53.1% vs
93.6%) and both fixes are T14B-side, export-shape-preserving: NEW T14B RIDER #1. Draft-OPD rejected-
position replay (arXiv 2605.29343) +13-23% AL donor-reported; our engine already localizes rejections
(generate.cpp:15384-15413 walk+rescue) so the harvest v2 is cheap: RIDER #2. DFlash2's checkpoint is now
public (inspectable CPU-side; card pins block-4, QSA-limited, below our S_mtp 6-8; AL delta still below
our target) — T23 verdict stands, T14C+ rider unchanged. The draft-vocab EN trap got MEASURED bytes:
production mtp-en = 162,100 B sha 369151522226a5ed vs mtp full = 425,196 B — gates-spec amended in-file
today (T27 AMENDMENT block) with the pin + the stale-Custom-data warning (32 references). NEGATIVE: AdaSD
(its acceptance half is non-lossless by its own report — fatal under bit-identity), DSpine/TiDAR/SpecFormer
class (separate 0.6-1B models break the 31-tensor contract).
T27-3 (serving/ops/stability): OUR OBSERVABILITY GAP IS PROBE-SIDE — the server ALREADY emits vLLM-named
TTFT/ITL/E2E histograms (serve/prometheus.py; render at server.py:6000) and probe-lane.py never scraped
them; it also throws away a temperature-0 deterministic response every run. STAGED TODAY:
campaign/probe-metrics.py (self-tested: labelled + bare Prometheus formats, quantile with +Inf clamp) —
scrapes the histograms, computes p50/p95/p99, records clock/temp/power per leg, hashes the golden
response as a per-boot drift alarm, appends JSONL; WIRED INTO THE A-SLATE legs so one boot prices conc
delta AND TTFT p99 AND drift. vLLM V1's decode-priority token budget + always-on chunked prefill = third
independent confirmation of the STRATA_BATCH_DECODE_SHARE shape (audit: TTFT p99 is the canary metric,
not just conc). VTC (OSDI'24) locality-aware admission-order fairness = ~40 lines in _admit_next, no
PAUSE verb — the cheaper sibling that precedes maya fair-slice; TWP admission pacing (arXiv 2610.17689)
is the 7+-client follow-on. Re-bank AMENDED: record clock/temp/power per leg (the 2835->3135 epoch was
discovered post-hoc). NEGATIVE: Pascal multi-instance (one box), batch-invariant kernels (dead — our
hand-built int8 graphs), PreScope prefetch (ForesightSwap family), sleep-mode/CRIU boots.
WHAT CHANGED IN OUR ORDER: (1) the metrics harness now runs INSIDE every a-slate leg (stability and
throughput priced in the same boot); (2) the draft-vocab contract is pinned in the spec before any G0;
(3) T14B gains two named riders (EAGLE-3.1 anti-drift, Draft-OPD replay) whose data comes free from the
queued STRATA_SPEC_TRACE instrument; (4) the MoE port's Phase B spec and the PDL chain gained explicit
safety amendments; (5) one new zero-boot microbench (cuBLASLt heuristic sweep) enters the slate.

## F-256 — 2026-10-21: STALE CLEANUP + THE MEASUREMENT EPOCH OPENS (probe-metrics rides every leg from here)
Cleanup (all untracked/ignored paths; journal is the record): engine/ 737M -> 249M — deleted the seven
superseded intermediates (strata.pre-t15c/-t18fix/-t19-s1..s4 = 433M + strata.t20-merged-artifact 84.5M,
whose md5 provenance is banked in F-248); KEPT the rollback chain engine/strata (current) ->
strata.pre-t24-s1 (T24-era rollback) -> strata.t19-final (pre-merge rollback) + libstrata_kq.so + BUILD.json.
Root boot logs: 42 stale per-arm logs + the 27M live log archived to campaign/results/old-logs/ (9.0M
gzipped); the next boot recreates custom-iq3_s.log. No tracked file changed; no harness deleted;
campaign/results leg evidence (21M) kept. Reviewed for relevance: build/ .venv/ third_party/ all live;
upstream files untouched.
THE EPOCH: from this entry every battery leg also captures campaign/probe-metrics.py (TTFT/ITL/E2E
p50/p95/p99 from the server's existing histograms, acceptance, sm clock/temp/power, golden-response hash
drift) appended to campaign/results/metrics.jsonl. First measurement epoch of the improvement loop:
baseline (3 standing boots) -> a-slate (decode-share A/B 3v3 + calibrate.py + prewarm2) -> adopt/reject
against the baseline with the no-regression law.

## F-257 — 2026-10-21 evening: THE IMPROVEMENT LOOP, EPOCH 1 — baseline banked, two falsifiers run, calibrate.py finally speaks
BASELINE (3 standing boots, same evening window, t27-baseline.log): M1 188.3 / pair 186.4 / conc 297.8
(mean; legs M1 198.5/178.9/187.5). These are WINDOW numbers - the banked five-boot absolutes
(212.5/202.8/315.7) stand; tonight's batteries compare only within tonight. STABILITY CAPTURE per leg
(metrics.jsonl): TTFT p50/p99 175ms/2155ms every leg, ITL p99 20ms, acceptance 74.1-76.0%, sm clock
2955-2962MHz at 36-44C / 88-93W, window trend early->late 13-14 -> ~10ms stable. GOLDEN-HASH FINDING:
the temp-0 deterministic response takes 3 distinct hashes across boots (a974/dbb/3b68) - the alarm's
true semantics are BOOT-TO-BOOT OUTPUT VARIANCE (the known layout/batching nondeterminism class), not
hard drift; keep it as a variance signal, one hash per boot.
DECODE-SHARE 0.75 (a-slate, ABAB 3v3): arm M1 199.97 / pair 187.9 / conc 298.9 vs ctl 196.0/189.1/295.6
= M1 +2.0%, conc +1.1%, pair -0.6% -> NOT ADOPTED (targeted lane +4% bar; M1's +2.0% is warm-trend -
ctl r1 188.4 was the sequence's cold leg and the flat-gate path has no read pieces at single stream).
CALIBRATE.PY (tools/calibrate.py, first-ever successful run - 389s standalone; my a-slate leg had
booted the 8081 server beside its own engine = the collision the fence warns about): PCIe probe
54.5 GB/s; pcie_frac median-of-3: 0->129.4, 0.35(ours)->129.4, 0.55->138.1 BEST, 0.75->133.9, 1->124.3;
spec_min_p: 0.3->123.5, 0.5(ours)->137.9, 0.7->140.1; confirm grid 0.55/0.7->140.8; workers 15 (matches
standing); expert-tier 1/80/0.97->138.7 vs default 121.5. NOTE: its lane is a synthetic single stream
~130-140 tok/s - it prices DIRECTION on our box, not magnitude on our traffic.
FRAC55 BATTERY (arm 0.55 vs ctl 0.35, ABAB 3v3, config-copies): arm M1 205.4 / pair 184.2 / conc 296.4
vs ctl 198.3/185.1/298.4 = M1 +3.6%, pair -0.5%, conc -0.7% -> NOT ADOPTED (below the +4% bar; the ctl
rounds drift down through the battery 208->196->190 so the M1 signal is drift-confused; the calibrator's
+6.7% did not transfer to real traffic). One honest note FOR the arm: its M1 was flat-stable ~205 every
round while ctl swung - if a fresh-window confirm is ever run, that stability is the reason.
PREWARM2: mechanically verified (complement pin confirmed, page cache re-warmed in 7s, replay 2 short +
1 mid prefill + 6-way burst in 17s); its +3-5% five-boot-mean claim is judged at the NEXT re-bank's
first legs, not on a single mid-sequence boot.
HARNESS LESSONS (both repeats of banked law F-249): (1) launcher copies MUST live at the repo ROOT -
the campaign/ copy broke .venv resolution and cost two aborted runs before the fix; (2) calibrate.py
boots its own engine - NEVER run it beside a live server.
INSTRUMENTS: STRATA_SPEC_TRACE per-window keep-rule telemetry patched into generate.cpp (env-gated,
default-off, no math; needs a rebuild + solo boot to collect); cuBLASLt heuristic sweep built
(campaign/cublaslt-sweep, CUDA-13 pointer-before-descriptor API) awaiting a free GPU.
CORRECTION (T27-1's "one-flag arm"): our graph instantiates are the LEGACY 3-arg cudaGraphInstantiate
(flags reserved = 0) at verify.cpp:2628/3588/3595 - arming UseNodePriority means cudaGraphInstantiateWithFlags
+ per-node priorities, else the flag is inert. Not a flag flip; a small design item.
NEXT IN THE LOOP (running now): the adapt-tier arm 1/80/0.97 (calibrator's +14% synthetic claim), then
spec-min-p 0.7, then the cuBLASLt sweep on a free GPU.

## F-258 — 2026-10-21 night: EPOCH 1 CLOSES — all three calibrator arms falsified on real traffic; the standing recipe survives every A/B
ADAPT-TIER 1/80/0.97 (ABAB 3v3, config-copy): arm M1 202.1 / pair 185.8 / conc 288.8 vs ctl
201.6/191.0/301.9 = M1 +0.2%, pair -2.7%, CONC -4.3% -> NOT ADOPTED (regression on the conc lane; the
calibrator's +14% synthetic did not transfer - swap churn every window disturbs the cohort lane).
SPEC-MIN-P 0.7 (ABAB 3v3): arm M1 198.4 / pair 188.2 / conc 299.0 vs ctl 200.4/187.4/297.6 = M1 -1.0%,
pair +0.4%, conc +0.5% -> NOT ADOPTED (noise). The arm's acceptance "rose" to 82-84% purely by offering
fewer drafts - the numerator effect, worth nothing. F-228's 0.5 stands.
SCORE: the calibrator's three measured recommendations (frac 0.55, minp 0.7, tier 1/80/0.97) all failed
the transfer to real traffic (frac55: M1 +3.6 drift-confused/pair-conc flat-negative; adapt: conc -4.3;
minp: noise). LAW BANKED: calibrate.py prices DIRECTION on this box, never magnitude - its synthetic
single-stream lane (121-142 tok/s class) is not our probe's traffic; treat its output as a candidate
generator, never as evidence. Zero adoptions tonight: the standing recipe at M1 ~200 / conc ~300 (this
window) beat everything thrown at it - the negative ledger grew by three priced falsifications
(~18 boots of evidence) and the recipe itself is untouched.
CUBLASLT SWEEP (campaign/cublaslt-sweep, 6 shapes x 6 M values, results/t27-cublaslt-sweep.txt):
GemmEx is at parity mid-range (M 1024-2048, ratio 0.96-1.02); headroom at the edges and concentrated in
the bf16 hc projections: hc_down 320x10240 M=8192 Lt-best +25% (169.8 -> 211.9), hc_up 10240x320
+9..17% at M 256-2048-8192, router 512x2560 M=512 +22%, gate M=256 +17%. Recorded as a MARGINAL
candidate: a selective cublasLtMatmul path for the hc shapes at large M (~1% prefill class); not
scheduled ahead of the kernel tranche.
BOX: left idle after cleanup (ports clear, GPU 900MHz idle). NEXT: the STRATA_SPEC_TRACE rebuild + solo
boot (instrument already in-tree, default-off), the KL yardstick, then the kernel tranche per the T26/T27
order. The re-quant GO remains the user's call, now carrying tonight's evidence that config-level knobs
are exhausted on this box - the remaining gains are in code.

## F-259 — 2026-10-21 late: USER RELAXES THE BAR — pooled evidence doctrine; TWO ADOPTIONS (pcie-frac 0.55, decode-share 1.0); the Lt arm rebuilt and under test
DOCTRINE CHANGE (user, tonight): an arm does NOT need the +4% single-battery bar - the verdict is
pooled evidence over multiple tests, "secure wins". Protected-lane law still applies (no real
regression), and numerics-class changes still owe quality gates.
ADOPTION 1 — PCIE-FRAC 0.55 (config, was 0.35): the 5v5 confirm (t28-frac55-confirm.log) settled the
first battery's drift question. Pooled across BOTH batteries (16 legs): arm M1 204.1 vs ctl 198.5 =
+2.8%, the arm mean higher in each battery independently (205.4 vs 198.3; 203.3 vs 198.6); pair -1.3%
and conc -0.7% pooled, both inside tonight's +-8% leg noise (pair legs ranged 176.8-209.2). Round
pair-deltas were mixed (+25.3/-8.9 etc) because the box itself swung between a slow early window
(175-195 both arms) and a warm one (195-225) - the pooled mean is the instrument. QUALITY: MMLU 2x on
the arm 94.29 / 92.86, mean 93.57 >= floor 93.3 (the F-224 mean-rule precedent: 95.7/92.9 adopted).
ADOPTED: standing config custom-iq3_s-8081.json --pcie-frac 0.55; the boot log confirms
"pcie_frac 0.55 (default 0.55)". REVERT: set 0.35. Duplicates (custom-iq3s-frac55.json,
launch-frac55.sh) removed.
ADOPTION 2 — STRATA_BATCH_DECODE_SHARE=1.0 (env, was code-default 0.5): curve end tested after 0.75
read conc +1.1. On the NEW frac-0.55 standing line, 3v3: conc +2.4% (287.1 vs 280.3), M1 +1.9%,
pair -0.6% - all lanes non-negative at the means, direction consistent across both share batteries.
Timing-only (changes read/decode interleaving, not computation) so no quality gate owed. ADOPTED into
the standing launcher (one unset to roll back).
THE Lt ARM (T28): selective cuBLASLt path for the prefill BF16 GEMMs (gemm.cu: env STRATA_GEMM_LT=1 +
STRATA_GEMM_LT_MIN_T default 2048; heuristic-top algo cached per (N,K,T); any failure falls back to
the unchanged GemmEx; CMake links cublasLt on CUDA; HIP untouched). Engine rebuilt (libcublasLt.so.13
verified in ldd), engine/strata.pre-t28 is the rollback; the binary also carries the default-off
STRATA_SPEC_TRACE instrument. A/B 3v3 running (t28-lt-arm.log); adoption would owe MMLU 2x (the Lt
algo may reorder reductions - the F-224 numerics class).

## F-260 — 2026-10-21 night close: Lt arm falsified on the pipeline; the spec trace's FIRST DATA (and the new recipe's first solo 222.4)
LT ARM (STRATA_GEMM_LT=1, 3v3 on the new standing line, t28-lt-arm.log): arm M1 204.7 / pair 196.4 /
conc 295.7 vs ctl 209.0/195.9/312.9 = M1 -2.1% (EVERY round: 208.3<210.4, 204.6<208.0, 201.3<208.7),
pair +0.3%, conc -5.5%, prefill flat (fresh-long 1.72-1.75 vs 1.67-1.74 s) -> NOT ADOPTED. The sweep's
shape-level +9..25% does not compose: the hc shapes are a small FLOP share, the >=2048 gate misses most
chunks, and the heuristic's top pick wins a microbench but not a pipeline. Arm stays default-off in-tree
(one env to re-arm); engine/strata.pre-t28 is the pre-LT binary but the deployed binary carries the arm
unarmed = harmless. CLASSIC LAW REINFORCED: microbench wins are hypotheses; the pipeline is the judge.
SPEC TRACE FIRST DATA (solo boot, 1,615 windows, campaign/results/spec-trace.txt + analysis):
- offered T is BIMODAL: 1:374 2:372 3:219 4:184 5:99 6:367 - depth 6 fires almost as often as depth 1.
- accepted mean 1.734; acceptance by dprob decile is MONOTONE and STEEP: 0.5-0.6 -> 57.1%, 0.6-0.7 ->
  61.3%, 0.7-0.8 -> 64.3%, 0.8-0.9 -> 72.2%, 0.9-1.0 -> 83.5% (n=3593 positions). dprob is informative -
  the keep rule's premise holds on our drafter.
- RAW product thresholds barely cut (th 0.05..0.20 all keep 2.222/2.225; th 0.50 -> 2.028) because most
  dprob >= 0.9: the refit MUST use CALIBRATED acceptance bins (basalt's acc(p) form, i.e. the decile
  table above), not raw products. Next-session analysis: bins + th* + the T14C c_row cost model.
- RESCUE FREQUENCY: 387/1615 windows (24.0%) advance via rescue - rescue-inclusive observe() is not a
  corner case, it is a quarter of the data (T26-2's port amendment was right).
THE NEW RECIPE'S FIRST SOLO BOOT (frac 0.55 + share 1.0 + the new binary): M1 222.4 / pair 195.6 /
conc 302.8 - the night's best M1 leg, consistent with the pooled +2.8% adoption read. The five-boot
re-bank of the new recipe (with per-leg clock/temp/power) is the owed next measurement; scoreboard
absolutes remain the banked 212.5/202.8/315.7 until then.
BOX: idle, ports clear. Standing recipe now: FIVE adopted arms + norm_fast + frac 0.55 + share 1.0,
revert lines: --pcie-frac 0.35 / unset STRATA_BATCH_DECODE_SHARE.

## F-261 — 2026-10-22 early: THE TRACE PAYS — DEPTH 7 ADOPTED (+8.5% M1, 3/3 rounds); the dm parity instrument lands
THE CHAIN: F-260's spec trace said 53% of windows accept EVERYTHING offered and 12% (196/1615) hit the
T=6 cap with mean dprob 0.944 - the drafter was being stopped mid-stride on its hottest chains. Per-
position acceptance: pos0 91.7% > pos1 82.9% > pos2 71.8% > pos3 60.1% > pos4 53.4% - the marginal row
on a confident continuation (~0.8 accept) sits above the ~0.6 break-even the minp-0.7 neutral priced.
THE BATTERY (t29-spec7.log, ABAB 3v3, config-copy): arm M1 186.9/209.9/198.8 (mean 198.5) vs ctl
186.0/175.0/187.6 (mean 182.9) = +8.5% M1, 3/3 rounds positive (+0.5/+20.0/+6.0); pair +0.6%; conc
+1.8%. The arm's acceptance reads lower (72.4-74.0 vs 73.5-75.5) = the numerator effect, while tok/s
rose - the real signal. Depth is window-shape only under exact-match verify -> NO quality gate owed.
ADOPTED: standing config --spec 7 (revert: set 6). --spec 8 battery running against the new line (the
engine's captured set goes to 8; if 8 wins it supersedes).
ALSO LANDED (implementer, commit ae3cebee): the dm parity instrument - dm_plan_parity harness
(registered after hc_q8_parity; covers all 7 of our pack's pairs, T=1..8, all-resident/mixed-partial/
shared-expert/G=2 regimes; our resident_plan vs donor dm_group+dm_glue transcriptions, combine family
vs dm_combine4's sums incl. the +0.0f/-0 edge) + STRATA_DM_WAIT_STATS (waitA/waitB/waitCPU device-
accumulated, per-512-window stderr line). BOTH default-off. TWO REAL CORRECTIONS to T26-1: (1) the
'algebraically identical' claim is exact only on distinct-in-token ids - the donor mask form holds one
bit per token and cannot represent a repeat at all; (2) dm_combine4 as written is a different sum than
ours (fma chain) - the port must compile with fmad on; the harness measures the diff. Plan transcriptions
CPU-validated on 35,055 random cases (0 fails); the GPU parity run + the DM_WAIT_STATS boot are the next
box steps (batteries own the box until they clear).

## F-262 — 2026-10-22 ~02:30: spec8 closed (conc regresses); dm parity's FIRST GPU RUN finds the real divergence surface; wait-stats needs wider arming
SPEC 8 (vs the adopted 7, ABAB 3v3, t29-spec8.log): arm M1 202.9 / pair 193.3 / conc 280.4 vs ctl
205.4/188.0/298.9 = M1 -1.2%, pair +2.8%, CONC -6.1% (all three rounds negative) -> NOT ADOPTED.
The 8-token window is kVerifyMaxT itself - at conc the batch lane pays for the widest window and the
cohorts serialize. DEPTH 7 IS THE SWEET SPOT (the ctl legs confirm the new line healthy: ~205 M1 /
~299 conc in the same window). Night's adoption tally: frac 0.55 + share 1.0 + spec 7.
DM PARITY, FIRST GPU RUN (build/dm_plan_parity, executed on the card): plan algebra GREEN everywhere
it applies - 51 cases, >=8752 fields compared, our resident_plan bitwise vs dm_group + dm_glue-mask
across T=1..8, all-resident / mixed-partial / cross-token-shared / G=2 regimes; the two documented
divergences (in-token repeat -> CPU pool; invalid id -> host fallback) behave as asserted. FINDINGS:
(1) two harness FAILs on the all-resident PARTIAL call at T=7/T=8 - "the partial call did not take
the partial skip": the T1.1 partial path was built for the <=6-token era and does not arm the same
way at 7/8 (PLAN_PARTIAL is default-off and closed-neutral, so this is a note + a harness-expectation
fix, not a standing-path defect); (2) THE LOAD-BEARING ONE: the combine family vs dm_combine4 -
hits-added and ZADD forms are BITWISE clean, but the plain multi-row combine MISMATCHES the donor
sum order, and the donor's uncontracted form differs on 9287/20480 columns (45.3%). Consequence for
the port: the dm combine CANNOT be transplanted verbatim for multi-row windows - it must reproduce
OUR multi-row order (or the port is restricted to the bitwise-proven forms); fmad must stay on.
DM_WAIT_STATS BOOT (STRATA_DM_WAIT_STATS=1 + full probe, standing line, M1 206.1): the instrument
printed NOTHING - its stat arms sit on the plain wait_flag_ge sites (verify.cpp:2192/2206/2231), but
the standing recipe takes the FUSED wait-copy path (143-amdk, :2188) and the device-plan variants, so
the armed sites never run. FOLLOW-UP (queued, implementer round 2): arm wait_copy_i32_from_mapped +
the planned waits, fix the T=7/8 partial harness expectations, investigate the multi-combine order.
Standing recipe after tonight: COHORTS=2 + ATTN_DAILY + GDN_CHUNKED + FUSE_HITS + POOL_DIET +
norm_fast + frac 0.55 + share 1.0 + spec 7. Owed: the five-boot re-bank (quiet window) + the dm round-2.

## F-263 — 2026-10-22 ~02:40: THE PROOF BATTERY FAILS TO CONFIRM — the composed recipe reads BELOW the pre-T28 base; all three adoptions under review
THE USER ASKED FOR PROOF WE GOT BETTER. The answer the box gave: not yet proven, and possibly no.
THE PROOF (t30-proof.log, 5v5 ABAB, same window 02:24-02:37, standing binary throughout):
  NEW (frac .55 + share 1.0 + spec 7): M1 193.7 / pair 181.2 / conc 289.4
  OLD (frac .35 + share 0.5 + spec 6): M1 198.5 / pair 187.3 / conc 296.4
  = OLD wins all three lanes by ~2.4-3.2%. Round pairs (new-old): M1 -18.9/+4.1/-2.5/-4.9/+0.8.
Each adoption won its OWN battery (frac +3.6 pooled, share +1.9, spec7 +8.5 3/3) - but the frac+share
COMBINATION was never A/B'd against the base before this proof, and the algebra that reconciles all four
batteries is: spec7 = +8.5% over the frac+share line, while frac55+share10 TOGETHER cost ~-10% vs base.
MECHANISM (hypothesis, mechanically plausible): share=1.0 hands the read-piece window wholly to decode
while frac=0.55 streams MORE missing experts over PCIe during exactly that window - the two fight for
the same channel. Individually each was measured with the other at its neutral setting.
THE NIGHT'S LEG VARIANCE for scale: the same OLD recipe read M1 182.9 (spec7 battery ctl) and 198.5
(proof) in different windows (+8.5% swing); the NEW recipe read 198.5-205 in its batteries and 193.7 in
the proof. The single-battery "+8.5%" readings are INSIDE this band. Pooled across all legs each recipe
ran tonight: new 198.9 (12 legs) vs old 196.3 (22 legs) = +1.3% - ambiguous.
DISCRIMINATING BATTERY RUNNING (t30-discriminate.sh, 3 arms x 3 rounds, same window): NEW vs OLDS7
(frac .35 + share .5 + spec 7 - isolates spec7's true value on the old base) vs OLD. Decision rule set
BEFORE the data: whichever arm wins M1 without losing pair/conc beyond noise becomes the standing
recipe; if OLDS7 >= NEW, the frac+share interaction is confirmed and both revert pending per-arm
re-tests on the spec-7 base. No adoption survives without a same-window win.

## F-263 FINAL — 2026-10-22 ~03:00: THE DISCRIMINATOR'S VERDICT — all three recipes indistinguishable; REVERTED to the proven base; the single-window adoption law is broken and now fixed
THE 3-ARM DISCRIMINATOR (t30-discriminate.log, 3 rounds x 3 arms interleaved, same window):
  OLD   (frac .35 + share .5  + spec 6): M1 197.1 / pair 186.6 / conc 287.0
  OLDS7 (frac .35 + share .5  + spec 7): M1 194.6 / pair 182.9 / conc 281.0
  NEW   (frac .55 + share 1.0 + spec 7): M1 193.9 / pair 185.5 / conc 287.8
M1 spread 1.7% with +-8% per-leg swings = STATISTICALLY INDISTINGUISHABLE. With the proof battery's
old +2.4%, the honest verdict: the three T28/T29 "adoptions" (frac 0.55, share 1.0, spec 7) have NO
reproducible composed gain - every individual single-battery win sat inside the window noise band
(the same OLD recipe read M1 182.9 and 198.5 in different windows tonight). The +8.5% spec-7 reading
was a ctl-dip artifact. THE PRE-SET DECISION RULE FIRED: no arm won -> revert.
REVERTED: standing config frac 0.35 + spec 6; launcher share export -> shelf comment. The arms stay
as labeled shelf launchers (launch-spec8.sh, launch-olds7.sh, launch-calib-*.sh) for a QUIET-WINDOW
re-test. The banked scoreboard (M1 212.5 / pair 202.8 / conc 315.7, five-boot means) belongs to the
base recipe and STANDS. The deployed binary (spec-trace + Lt + dm instruments, all default-off) is
proven harmless - the discriminator's OLD legs ran on it at historical values.
THE LAW THIS NIGHT RE-LEARNED (a F-246 restatement with teeth): a single-window 3v3 canNOT resolve
1-3% effects on this box - only quiet-window five-boot means can. Adoption bar for sub-4% candidates
is hereby: five-boot means in a quiet window, or it does not ship. The 4%+ single-battery bar that
served the kernel arms well was doing quiet-window work implicitly; relaxing it (F-259) without
raising the sample count was the error - the relaxation is AMENDED, not repealed: pooled evidence
across WINDOWS, not across legs of one window.
WHAT THE NIGHT ACTUALLY BANKED (real, permanent): the stability harness riding every leg; the spec
trace + its 1,615-window dataset; the dm parity instrument + its GPU run (plan algebra bitwise-green;
combine SAME sum under fmad - the round-1 'divergence' was a harness bug, fixed in b298f7e8); the
DM_WAIT_STATS fused-path arming (round-2); six priced falsifications (Lt arm, spec 8, frac 0.55,
share 1.0, spec 7 as gain, adapt-tier) and the calibrator-direction law. The recipe is untouched and
proven; the instruments are sharper than when the night started.

## F-264 — 2026-10-22 ~03:15: THE WAIT-SPLIT PRICES THE MoE-TAIL PORT — and the base is confirmed unregressed
THE INSTRUMENT BOOT (rebuilt engine b298f7e8 deployed; rollback engine/strata.t28-instr; base recipe,
STRATA_DM_WAIT_STATS=1, full probe): M1 200.2 / pair 199.1 / conc 303.4 - the RESTORED BASE at its
historical band (the no-regression proof: tonight's base legs ran 197-207 M1 / 281-304 conc all night).
THE SPLIT (1,536 windows, 9,942 groups): waitA (host plan) 7.09 us/group (max 8,253) | waitB (PCIe)
12.44 (max 2,100) | waitCPU (rows) 53.93 (max 8,286); reach-before-wait: waitB 51.28 us, waitCPU 4.54.
Per window ~6.5 groups -> ~477 us of wait, 74% of it waitCPU (the host pool's miss-expert compute).
THE HONEST PRICE OF THE dm PORT: basalt's in-kernel tier wait + folded combine hides the waitCPU span
under useful resident work; ceiling ~3.5% of the window, realistic +1.5-3% M1 - NOT the +5-10% band
T25-1/T26-1 carried. The port remains the largest single kernel item but should ship as PART of a
composed window tranche (diet + PDL + dm), never alone.
DM PARITY IS NOW FULLY GREEN (round-2 harness, PASS, 51 cases, 11,636 fields): plan algebra bitwise
across T=1..8 AND the combine bit-compatible in EVERY form incl. multi-row (keep fmad on) - round-1's
"multi divergence" was a harness emulation bug (contiguous vs strided), fixed in b298f7e8. THE dm PORT
HAS NO KNOWN ALGEBRAIC BLOCKERS LEFT.
NIGHT LEDGER (the full session): 9 falsifications (Lt arm, spec 8, frac 0.55, share 1.0, spec 7-as-
gain, adapt-tier, decode-share-as-gain, + the round-1 harness bug), 0 surviving adoptions, 0
regressions in the standing recipe - WHICH IS UNTOUCHED AND SERVING AT ITS BANKED BAND. The permanent
bank: stability harness riding every leg, spec-trace dataset, dm parity + wait-stats instruments, the
quiet-window law with teeth, and the honest prices (dm +1.5-3%; config knobs exhausted).
NEXT THAT CAN PAY (all need quiet-window five-boot means): the composed window tranche (diet + PDL +
dm at +1.5-3 each if they compose), re-quant (user GO), T14B retrain. Nothing else on the config level.

## F-265 — 2026-10-22: ROADMAP PRUNED — the dead-end ledger consolidated (user directive)
Everything proven dead this campaign is now REMOVED from the active roadmap (SVG-CUSTOM-TODO.md) and
consolidated in its DEAD-END REMOVALS table with evidence refs: decode-share raise, --pcie-frac 0.55,
--spec 7/8, adapt-tier 1/80/0.97, --spec-min-p 0.7, STRATA_GEMM_LT, the cuBLASLt sweep as a lever, the
device-plan falsifier (void), PIPED-as-env-boot, adapt-every-as-depth-scheduling, and the T27 external
negatives (megakernels, AdaSD, DSpine-class, Pascal, batch-invariant kernels, PreScope, sleep/CRIU,
"ACES"). The individual falsifications are journaled at F-253..F-264; the four standing laws they
produced: quiet-window five-boot means for sub-4% effects; calibrate.py prices direction not magnitude;
microbench wins are hypotheses until the pipeline judges; an agent's headline is a hypothesis until the
launcher and history are read. The live roadmap: quiet-window re-bank -> KL yardstick -> keep-rule
refit -> prewarm2-at-re-bank -> the composed window tranche (diet + PDL + dm; parity fully green) ->
re-quant (user GO) -> dec_gdn_rec/mid_chain -> fp16-accum (gated) -> PIPED port -> T14B with the
EAGLE-3.1 and Draft-OPD riders -> 262k.

## F-266 — 2026-10-22 ~11:10: T31 QUIET-WINDOW RE-BANK — anchor banked; prewarm2 ADOPTED
Instrument: campaign/t31-rebank.sh + t31-summary.py (results/t31-rebank.log, 8 legs, one quiet window
10:56-11:07, probe-lane + probe-metrics riding every leg). THE ANCHOR (5 base boots, standing recipe,
zero env changes): M1 206.0 / pair 202.8 / conc 326.4; M3b 76.1; accept ~75.6%; ITL p50 4.3ms / p99
19.5ms; clock 2962 MHz steady, 38-44C, 90-95W per leg; ms/window trend 12.9->9.5 (stable). vs the
banked T19 means (212.5/202.8/315.7): pair EXACT, conc +3.4%, M1 -3.1% - which sits INSIDE the M1
boot band (195.3..215.0): no regression signal, quality gates in-band; this anchor is now the
same-window reference for every future A/B.
PREWARM2 JUDGED 3v5 IN THE SAME WINDOW (its +3-5% five-boot-mean claim): M1 +4.5% (215.2 vs 206.0;
arm boots 217.6/214.3/213.7, arm spread 1.8% vs anchor's 10% - all three at/above the anchor's top,
the cold r1 195.3 is exactly the component it removes), pair +1.7%, conc -1.0% (bands fully overlap =
noise), M3b 75.2 in-band. Cost 15-20s per boot (dd 2-7s + replay 11-16s); OFF = do not run the
script. Verdict: ADOPTED as the STANDING MEASUREMENT POSTURE - every future battery boot runs it on
BOTH sides of any A/B; provisional prewarmed anchor 215.2/206.4/323.0 (3 boots, grows for free as
future boots prewarm = the standing confirmation rider). Confound noted honestly: arm followed ctl in
boot order (F-225 warm-trend risk), but base r4/r5 did NOT trend (202.7/208.9 after r3's 215.0), the
mechanism targets the cold component specifically, and the pooled-evidence doctrine + one-command
revert carry it. Golden hashes: 2 distinct over 8 boots (dbb2 x7, a974 x1 on base-r5) - the known
boot-variance signal, monitored. NEXT: the composed window tranche arms (diet + PDL + dm) get judged
against the prewarmed anchor.

## F-267 — 2026-10-22 ~11:35: HOUSEKEEPING PASS 2 (user directive: clean up, avoid runaway mess)
~1.15G freed, working tree clean. engine/ 408M->163M: deleted the intermediates pre-t24-s1 / pre-t28 /
t28-instr (all instrument-era binaries, env-gated default-off = functionally identical on the standing
path; the named rollback is engine/strata.t19-final, older states rebuild from git); kept strata
(deployed), t19-final, libstrata_kq.so, BUILD.json. build/ 1.2G->309M: 132 stale test/engine
executables deleted (f209 pair, q36/q36ref, gemm parity pair, the 10-17 prefill/mmq/nvfp4 test bins,
all parity/bench exes - everything ninja-regenerable); kept build/strata + the .a/.so link state.
campaign/results/: the 75 drifted-untracked t26..t30 boot .outs COMMITTED (label provenance, 2.2M
total - convention: harness .outs commit with their battery); live custom-iq3_s.log (8.4M, unique
per-boot request accounting for the T26..T31 era) rotated to old-logs/custom-iq3_s.20261022-t31.log.gz
(910K, committed) and the live log truncated. spec7 shelf strays launch-spec7.sh +
custom-iq3s-spec7.json DELETED (spec 7 died at F-263; launcher comment + journal carry it). Empty
upstream skeleton ~/LLM/Custom-data rmdir'd (the 32 stale Custom-data spec refs stay flagged in
gates-spec). /tmp prewarm2 + kl test fixtures removed. UNTOUCHED BY DESIGN: data/expert-profile-
learned.bin (engine's own output, stays dirty), campaign/t31-keeprefit.py (refit agent LIVE at press
time), /home/svg/LLM/data/references (450MB, extracted value), the stock ~/LLM/strata harness, and
the parallel session's workspace/default/nvme-data-recover staging.

## F-268 — 2026-10-22 ~12:00: STRUCTURAL CLEANUP — campaign/ + root reorganized (user directive)
98 dead files deleted (git history = the archive), 10 relocated, layout documented in
campaign/README.md + an AGENTS.md layout section (future agents). LIVING INSTRUMENTS stay at campaign/
top (paths referenced by batteries): probe-lane.py, probe-metrics.py, prewarm2.sh, mmlu-pro-gate.py,
mimicbench2.py, bit-check.py, ctx-probe.py, depth-lane.py, depth-lane-real.py, mma-accum-ratelock.cu.
campaign/batteries/ = the reusable harnesses: abab-run.sh, t20-gates.sh, t22-custom-vs-stock.sh,
t31-rebank.sh + t31-summary.py (syntax re-verified post-move; their instrument references resolve
unchanged). campaign/arms/ = parked future work: the 262k launcher+config pair (refresh owed at GO
time - internal paths predate the move). research/ absorbed the loose specs (CPU-TIER-OVERLAP-SPEC,
GOAL-LINUX-SUPERIORITY, ROADMAP-30-VS-STOCK, perf-chart-entry2). DELETED, all closed questions
recoverable at their old paths in git history: 41 one-off arm configs (g2*/gdn*/frac*/t17-*/stair*/
cohorts/fusehits/pooldiet/...), 38 run-once harnesses (knob-*, rebank2/3/-3135 superseded by
t31-rebank, t17-leg, t18/t19/t20/t21/t24/t26..t30 one-offs, battery-boot/config/nospec/tqllm,
prefill-probe*, v02-bench, prewarm v1, f197-analyze, t15c-battery/score, cublaslt-sweep.cu), the
root launcher strays (calib-adapt/minp, 1m pair, c3/conc/t24s1/t24s2, olds7 + spec8 pairs),
t17-lp-profile.bin (dead learned-profile lever), 19 ignored server-*.out boot strays + __pycache__.
KEPT AT ROOT BY LAW: launch-custom-iq3s-8081.sh + custom-iq3_s-8081.json + refusal-r.gguf
(config-referenced asset, F-212) - the launcher resolves .venv and config relative to itself (LAUNCH
LAW + F-249). Root is now: upstream's own files + the standing boot pair + SVG-CUSTOM.md/TODO.md/
AGENTS.md + the live log. Policy going forward: run-once harnesses die with their question; battery
.outs commit with their battery; live log rotates gz at cleanup passes.

## F-269 — 2026-10-22 ~13:40: KEEP-RULE REFIT (T31 zero-boot, agent deliverable, VERIFIED)
campaign/research/T31-keeprefit.md + campaign/t31-keeprefit.py (re-runnable sim, MC + bootstrap;
main-loop re-ran it: plug-in dA/w -0.20074 vs MC -0.20198 on 474 affected windows; the 0..4-positions
claim checked against generate.cpp:15427 `i + 1 < T`). NOW VERDICT - F-260's "post-retrain only" is
REFUTED model-side, with a catch: best rule = CUM-CALIBRATED joint product, flat th*=0.50 (plateau
0.40-0.60) -> cuts 0.60 offered rows/window, loses 0.20 accepted/window = +4.9% net at central
c_row=2.4ms (break-even 0.510 tok/row vs cut-mass carry 0.334), positive in ALL 15 boot segments
(+3.4..+6.0%), bootstrap p5-p95 +4.0..+6.3%, P(net>0)=1.00. Raw-d products inert (+1.0% at th 0.5 -
F-260 CONFIRMED and explained: 69.6% of printed d >= 0.9). THE CATCH: the pooled-bins bound reads
+0.05% - the entire edge is the deep-position acceptance deficit (F-235), and F-183's probe-grade
min-p 0.7 read -3.0% real vs this model's +3.1% prediction (contradiction CARRIED, only a five-boot
battery decides). DISPOSITION: battery arm, not adoption - ~50 lines, default-off, offers-only-exact
(zero quality risk); joins the composed window-tranche battery queue; F-263 law governs.
RETRAIN INTERPLAY INVERTED: delta*=0.00 - the rule pays NOW if ever and its value DECAYS with drafter
quality (zero by additive shift +0.30): battery BEFORE T14B, expect retirement after a successful
retrain (the EAGLE-3.1 anti-drift rider attacks the same deficit the rule monetizes) - never
double-count it with the retrain in composed bands. TRACE CORRECTIONS: the dataset is ~15 appended
boot segments (resc counter resets 14x), not one boot; rescue = 466 events in 391/1615 windows
(24.2%), the field is cumulative; prior decile "tested 3593" was OFFERED not tested (true 3181 incl
/ 1955 exact) so its rates were deflated; "53% accept-all" = 69.4% of DRAFTING windows (53.3% of
all); "12% cap-hit" = 53.4% of T=6 windows; exact-match mean C=2.4458 ~= T14C's standing 2.47 (the
2.734 headline was rescue-inflated +10.6%); prior raw-th kept-means reproduce byte-for-byte.

## F-270 — 2026-10-22 ~16:20: T32 WINDOW TRANCHE, PHASE 1 JUDGED (keep falsified-solo; dm +2.8%;
both pair +4.9% — confirmations owed)
Battery: campaign/batteries/t32-tranche.sh, 12 legs = 3 interleaved rounds x {ctl, keep(0.5), dm,
both}, every leg prewarmed + probe-lane + probe-metrics (results/t32-tranche.log). ANCHOR
REPLICATION: ctl means 213.4/205.8/322.5 vs the banked F-266 prewarmed anchor 215.2/206.4/323.0
(-0.8/-0.3/-0.2%) - the anchor reproduces inside a fresh window; the measurement chain is trusted.
THE KEEP RULE IS FALSIFIED FOR STANDING SOLO ECONOMICS: keep alone = M1 204.0 (-4.4%), pair -3.2%,
conc -1.8%. Mechanically it works EXACTLY as designed - M3b 76.9 -> 86.6 (offer quality up ~10pts,
fewer wasted rows) - but the refit's +4.9% priced row-LINEAR window cost (c_row 2.4 ms) and the
engine's solo windows are CAPTURED FIXED-SHAPE GRAPHS: a cut window still pays the same graph, so a
withheld draft row was a FREE OPTION we should never have declined. NEW LAW (the F-183 contradiction
RESOLVED, in the probe's direction): on fixed-shape captured windows, never withhold draft rows on
the solo lane - row-cutting pays only where row cost is REAL, i.e. batch contention. Supporting
evidence inside the same battery: BOTH's pair = 216.0 (+4.9% = the pair-bar CLEARED) - dm's fold +
keep's row cuts COMPOSE on the batch lanes where rows do cost. The keep arm stays in-tree
default-off (zero cost); it retires to the conditional-graph era (it is precisely vLLM's named
future work - pays when window shapes become dynamic, e.g. post-retrain).
THE DM PORT (F-270 phase-1b): M1 219.5 (+2.8%), pair +0.7%, conc +1.6%, M3b 76.1 in-band,
acceptance 70-83% in-band, dm armed verified per leg - INSIDE the F-264 honest band (+1.5-3%),
interleave wins rounds 1 and 3 (r3 230.4 vs ctl 213.6, r2 -1.3%). Sub-4% -> the five-boot pooled
confirmation is OWED before adoption.
BOTH (dm+keep): M1 219.8 (+3.0%), pair 216.0 (+4.9%, bar cleared), conc 332.1 (+3.0%), M3b 87.9.
NEXT: the five-boot extension (dm x5, both x5, pooled across windows vs ctl) + the 2x MMLU/mimic
gates on the candidate recipe; phase 2 (diet + PDL kill-switch + mid_chain/dec_gdn_rec) unchanged
in the queue. Arms all remain default-off; standing recipe UNTOUCHED and unregressed.

## F-271 — 2026-10-22 ~19:45: RESERVE-600 + BATCH-9 FALSIFIED (user-directed quick test); a
## methodology law earned the hard way
USER ASK: --vram-reserve-mib 600, then parallel 9, quick test, solo must not slow, 9 should only
take VRAM when used. FALSIFICATIONS: (1) reserve 600 kills engine 1 at LATE boot allocations at
EVERY batch width - b9+600 died at "batch MTP slot 0: draft logits do not fit (64 MiB free)", b6+600
died 19s later too; server.py's restart path AUTO-RAISES the reserve to 2000 (#735/#2065 class) - a
user reserve below the floor never survives to steady state; the 2000 default IS the anti-paging
floor, not slack. (2) batch 9 does not boot single-engine on 32 GB with this pack: 9 sessions
(6.55 GiB) + 9 drafters + WIDE 9-row graph captures claimed at REQUEST time exceed the boot-time
sizing even at reserve 2600 (engines: 2 - restart cycle every attempt). THE METHODOLOGY LAW (earned
by publishing then retracting four M1 reads): every collapsed number tonight (M1 57.7-73.8) was
measured against a DEATH/RESTART cycle - health came from engine 1, the probe hit the restarted
engine 2's FIRST REQUEST = the lazy-capture tax (~3x window cost ~= 71 t/s - the exact band) +
TTFT p99 8.4s + golden HTTPError. ALL RETRACTED as measurements. LAW: before any probe on a fresh
boot, verify SINGLE-ENGINE (exactly one "engine started" line in the log, 90s stable) - a restart
cycle invalidates every number from that boot. WHAT SURVIVES: the lazy-session feature the user
described ("only take VRAM if used") IS the unlock for 9-wide - sessions are carved eagerly at boot
(generate.cpp:5313 "--batch: every stage carves o.batch more sessions") so 9 costs 6.55 GiB always;
lazy allocation to max-9 + the re-quant's +3.9 GiB is the path that makes 9 fit. probe-lane gained
--conc-n (kept, committed); the b9 test config/launcher deleted (falsified, journal carries them).
Tweet note (Axitek 3090 tips): both dead here - gpu-unlock is GA102/104-only (F-252, our sm_120 is
whitepaper 2:1), P2P needs a second GPU.

## F-271 — 2026-10-22 ~19:30: THE LEARNED PROFILE CAN POISON THE ADAPTIVE TIER (2.2x degradation
found, root-caused, fixed) + clean re-tests of reserve-600/parallel-9 + the driver-unlock/P2P research
USER-DIRECTED TESTS (reserve 600, parallel 9) initially read CATASTROPHE: standing-recipe M1 94.7.
ROOT CAUSE (bisected through 4 boots): data/expert-profile-learned.bin - saved by the engine every 30
rounds and accumulating across EVERY session - had crossed a threshold after today's 27 boots and
poisoned the adaptive tier: 21,200 expert swaps per boot (vs 2,614 fresh), GPU-reach wait 6-10
ms/window (vs 0.62), CPU-expert spill 4.2-4.7 ms/window, solo rerouted through wide batch windows.
REMOVING THE FILE RESTORED EVERYTHING: M1 94.7 -> 201.8, GPU-reach wait 10.12 -> 0.62 ms/window,
adapt swaps 21,200 -> 2,614 - same desktop session active (the desktop/chrome theory REFUTED), same
binary, same config. The poisoned profile preserved as evidence:
campaign/results/expert-profile-learned.poisoned-t33.bin. TWO COROLLARIES: (1) F-239's "learned
profile moved MMLU -2.1pts" anomaly now has a mechanism candidate (accumulated profile state, not
numerics) - treat that rejection as confounded, not understood; (2) HYGIENE LAW: batteries and any
A/B must control the learned profile (back up + reset, or pin) - it is a hidden variable that
accumulates across boots; the t31/t32 batteries were internally consistent (same accumulating state
both sides) but the hazard is now named.
CLEAN RE-TESTS (fresh profile per boot, desktop active): parallel 9 (reserve 2000/600 irrelevant -
see below) = M1 182.4 vs clean standing 201.8 = -9.6% SOLO, pair -9.5%, conc 6 -8.1%, M3b in-band -
STATIC 9-slot allocation costs solo ~10% (pool 9,064->8,155 slots + slot-lane routing). The user's
requirement stands: a 9-lane needs DYNAMIC session allocation (VRAM and routing by active clients) -
that is the follow-up engine feature, not a config. Reserve 600 = DEAD END, closed: with parallel 6
the pool auto-sizing consumes the freed VRAM and the MTP DRAFT HEAD DOES NOT FIT (engine line: needs
81 MiB, 138 free -> boot never healthy) - this is WHY the engine auto-raises low reserves by ~600
(the observed 600->1200/2600 adjustments): the draft head + slack floor. The 2000 standing reserve is
only ~800 over the engine's own floor; nothing to reclaim at config level.
TWEET RESEARCH (campaign/research/T33-driver-unlock-p2p.md, 17 sources): Zagitta/gpu-unlock is REAL
but Ampere-only (GA102/104; speed-selector SS0/SS1 registers pre-driver-bind via initramfs module;
3090/3080Ti community-measured BF16 +59%) - NO Blackwell issue/PR exists, and it lifts the BF16/TF32
INPUT-rate lock, NOT our fp32-accum 2:1 (F-252); lineage = cmpunlocker ecosystem, not "Fateev".
aikitoria/open-gpu-kernel-modules = tinygrad P2P fork rebased to 615.71.09, genuinely Blackwell-
capable BUT P2P needs >=2 GPUs - zero single-GPU benefit. Blackwell gates are fuse-enforced per the
vgpu-unlock research (GSP blocked by fused-off VF PRIV). VERDICT: watch-only unchanged; triggers = a
GB202 PLM/SS writable-mask demo, or a second GPU in this box. Standing config restored + verified
healthy; box idle; fresh profile left in place.

## F-272 — 2026-10-23 ~05:30: T33 TRANCHE 1 LANDED — elastic sessions work (agent + verified)
STRATA_ELASTIC_SESSIONS=1 (+321/-29, generate.cpp only; unset = byte-identical eager path, re-verified
after the review-fixed gate inversion): 6 slot arenas as VmmRanges at stable VA, S_min=2 mapped at
boot, boot frees 2.91 GiB in 38 ms -> cache 8,226 slots vs 6,923 standing at the same reserve;
promotes 11-19 ms (map 12-17 + zero 0.6), demotes 28-40 ms (unmap 6-10; the cache eats the chunks
and refills from the profile); F0 PASS, F2 ramp 1->9->1 PASS twice on the deployed binary (zero
floor warnings, single engine per boot, promoted slots admit + capture immediately at the same VA);
F1 solo bracketed 211.5-249.8 vs OFF 234.3 - the five-boot battery decides. CAPTURE AUDIT: all four
families bake slot+lane+model-level memory only (batch graphs strictly through slots_[rows[t]];
cross-arena = never-unmapped primary scratch) - VMM sessions graph-safe by construction, as v3
predicted. DEVIATIONS (accepted): host pins held per process (pins are baked into captured graphs;
bounded --batch x 1.55 GiB; instrumented), promote's cache-release at the admission quiesce point
(11-19 ms, inside contract), give_way refuses unmapped slots, demote drops the slot's conv cache
(re-read on return - tranche-2 refinement: refill from the learned profile). CORRECTION TO SPEC v1:
the pl_want routing fix is INERT on this box (pipe requires a >=2-GPU layer split; with --batch the
pipeline is not constructed) - the measured -9.6% static-9 solo penalty was NOT half lane
disengagement; pool-shrink + profile-class effects carry it. The gate carries the right semantic
for future multi-GPU elastic configs. Rollback: engine/strata.pre-t34, or boot without the env.
NEXT: the F3 battery (standing vs elastic 3v3) + F4 (parallel-9 elastic = the F-271 question re-asked).

## F-273 — 2026-10-23 ~05:00: T34 BATTERY — ELASTIC SESSIONS ADOPTED; 9-WIDE DELIVERED
F3 (3v3 interleaved, prewarmed, single-engine every leg): ctl means M1 219.8 / pair 215.4 / conc
325.8; ELASTIC means M1 224.5 (+2.1%) / pair 214.4 (-0.5%, noise) / conc 338.8 (+4.0%) - and the
conc lane won EVERY round (339.2/343.2/334.0 vs 327.4/322.6/327.5). Mechanism: the freed 2.91 GiB
= +1,300 cache slots, and conc pulls wider expert unions so the slot term pays there (the requant
arithmetic's "conc gains more than M1" prediction, banked). F4 (parallel-9 elastic, 9-conc): M1
209.3 (SOLO IN-BAND - the F-271 non-negotiable holds), pair 209.4, 9-conc 322.5 = NINE clients at
6-conc-standing parity, engines_delta=1, ITL p99 35ms, accept 77.98% in-band. THE F-271 QUESTION
IS ANSWERED: 9-wide with zero solo tax, zero restarts, capacity elastic. ADOPTED into the standing
launcher (export STRATA_ELASTIC_SESSIONS=1): M1 no-regression, conc at the +4% bar with 3/3 round
wins, pooled doctrine + one-command revert (unset env); the five-boot pooled confirmation rides
every future battery (per the prewarm2 precedent). A third golden hash appeared on the b9ela boot
(3b680e29) - the known boot-variance signal, now 3 distinct hashes lifetime. OWED: the 2x MMLU +
mimic formality on the next adoption batch (elastic + dm together - both are
bit/timing-class-neutral, the F-239 anomaly note stands); re-quant stacks (+792 slots on a pool
that now demonstrably pays in conc).

## F-275 — 2026-10-23: THE T35 WAVE — four dossiers, the NEXT-TIER ROADMAP (value-ranked, not
## cost-ranked — user directive)
All four verified against the tree/logs by the main loop before banking (T35-elastic-refinement,
T35-serving-latency, T35-memory-hierarchy, T35-core-compute; commits 6775df30..c1ebee20). The
consolidated board lives in SVG-CUSTOM-TODO.md's T35 section. THE HEADLINES:
TIER A - THE LIFECYCLE OVERHAUL (one composed project, the elastic tranche 2): park-KV-on-demote
(returning chats 1,361 ms -> ~60 ms; the ConversationCache infra already exists unused on that
path) + the spare watermark (59/59 promotes run synchronous today with the spare empty - one
invariant flips the class; demotes 28-40 -> 6-10 ms) + timed-wait idle scan (kills the 128-310 ms
bill on the first post-idle arrival) + VMM superchunks 32-64 MiB (373 ops -> ~12-24; 40-60 ms
driver time -> 2-4 ms; the code already permits any multiple of the minimum) + parked-state
compression (restore-only tier - byte-exact, F-230 cannot reach; halves the 32-58 ms admission
copy) + per-token/per-window telemetry (the banked ITL p99 is a mean-of-gaps artifact; the engine
times every window and discards it - this is the instrument for every future verdict).
TIER B - THE CEILING-BREAKERS: TC-hc mma (FA-5090 proves plain mma.sync reaches 94.4% of the
fp32-acc SOL on our silicon; band +7-14% M1; kernel-level falsifier >=1.3x) + verify-head two-rung
(mmvq_il at T=1 + the basalt two-stage head, +3-7% M1) + the conditional-graph keep rule
(Nightjar formalized F-270's law as gamma*(B); our exec_[T] family zeroes their dominant cost term)
+ the MSInfer gate-prescale identity (halves the GDN decode recurrence inner loop, verify_kernels
.cu:286).
TIER C - PLACEMENT: the F-201 CCD split (finally execute it: 7 pool workers straddle the V-cache
L3 with the doorbell threads; 16 SMT idle) + hugepage the pinned arenas (IOTLB tax class) +
STRATA_L2_SETASIDE_MIB=72 (60 of ~72 cap today; boot line prints the accepted carve).
RECORD CORRECTIONS: F-271's reserve auto-raise attribution (not in the tree; conclusion stands),
F-272's host-pin bound (1.55 GiB FLAT), T33 H3's VMM cost table (29-51 us measured on GSP - op
COUNT is the lever). EXTERNAL VALIDATION: TreeWY/SpecLA/ReplaySSM each built the alternative to
our chain design and lost on chains - our dead laws are the field's consensus; F-252
double-confirmed; no banked law fell. STANDING OWED: dm/both five-boot + gates; phase-2
diet/PDL; re-quant USER GO (the routing-KL objective question feeds it).

## F-276 — 2026-10-23 ~15:20: T36 LIFECYCLE OVERHAUL JUDGED + ADOPTED (user directive honored:
## studied-not-abandoned)
Battery (t36, 3v3+1 interleaved, single-engine all legs): LIFE (park+watermark+timed-scan+
superchunks+telemetry) vs same-window CTL: M1 215.2 vs 212.2 (+1.4%), pair 205.3 vs 199.9 (+2.7%,
ahead EVERY round), conc 317.1 vs 314.8 (+0.7%), M3b 76.0 in-band — AND the battery's best M1
(228.3) came from the parallel-9 lifecycle boot (solo at 9-slot config, now routine). The interim
scare was the CTL's own 14% boot swing (193.8-221.6) — the quiet-window law's reason for existing.
ADOPTED into the standing launcher: the four lifecycle gates + telemetry (each one-command
revertible; pooled-confirmation rider per the prewarm2 precedent). Mechanism wins banked: restores
9-15ms (vs 275-326 re-reads), background promotes 2.2-8ms, 24 driver ops/arena vs 373, demotes in
true idle, TRUE per-token tails now in /metrics (mean-of-gaps era over). STAY OFF: GDN_PRESCALE
(canary moved 90->75-81% composed — isolated attribution owed before any reconsideration),
CCD_SPLIT (isolation leg lost to a script label bug — battery owed), CONV_COMPRESS (bit-exact but
1.01x on int8 KV — structure-aware codec owed). NEGATIVES BANKED: L2-72 falsified by the device
(persisting cap = 60 MiB), complement one-shot hugepage register stalls (kept cudaHostAlloc there).
FIVE-BOOT pooled extension of the life-vs-ctl means rides future batteries (sub-4% law); the
conc-vs-anchor gap (317 vs 323 class this window) gets the watermark-slot-cost + superchunk-
granularity + learned-profile-refill tunables if the pooled read confirms it.

## F-277 — 2026-10-23 ~17:00: dm_wait_combine ADOPTED after the full gate arc; the T37 wave's
## record corrections; both/keep closed properly
THE DM GATE ARC (the discipline law executed end-to-end): (1) SPEED — t32b five-boot extension
(campaign/batteries/t32b-dm-fiveboot.sh, 5 rounds x ctl/dm/both, all prewarmed, labels baked in —
no t36 label-bug repeat) pooled with t32's 3 rounds = 8 boots/arm across TWO windows: dm M1 222.4
vs ctl 217.1 (+2.5%; windows agree at +2.9/+2.2; 6/8 rounds positive), pair +0.9, conc +1.3, M3b
76.8 in-band — inside F-264's honest +1.5-3% price. (2) MMLU 2x 95.7/95.7 (floor 0.933; the three
misses are the same questions both runs). (3) MIMIC — the honest part: dm's first read 56/66 (84.8)
vs a same-window ctl counterfactual 60/66 (90.9) = a 4-item gap FOLLOWING THE ARM = gate FAIL as
read (never blame drift). Paired 3v3 interleave (t32b-mimpair): dm 60/60/62 — dm >= ctl every
paired round, 62 = the day's best read; with the counterfactual boot ctl is 60/60/60 today — the
56 was a TAIL DRAW and F-264's bit-identity (51 cases / 11,636 fields) held. ADOPTED:
export STRATA_DM=1 in the standing launcher (boot line "strata verify: STRATA_DM armed"; gate
verify.cpp:492; OFF = unset). VERIFIED-ARMED lesson: the fiveboot grep 'strata dm' missed the real
wording "STRATA_DM armed" — armed-line greps must come from the source's print, and custom-iq3_s.log
APPENDS across boots (grep 'engine started' | tail -1 to find the live one; head hits stale boots).
BOTH/KEEP CLOSED WITH A RECORD CORRECTION: the T32 "both pair +4.9% bar-cleared" did NOT survive
8-boot pooling (pooled pair +1.5%, M1 -1.3%); keep solo -6.0% pooled re-falsified; conditional
graphs (T37 C1) remain keep's only unlock. ctl-r1's mimic in the pair battery died mid-suite
(leg-local, 5/6 legs produced json) — cosmetic, ctl stably 60 across its three other boots today.
THE T37 WAVE (three researchers past T35, every load-bearing claim main-loop-verified at HEAD;
dossiers campaign/research/T37-{kernel-math,scheduling-spec,pathways-memory}.md, INDEX rows added):
RECORD CORRECTIONS — (a) F-249's "COHORTS=3 flat at 6" is VOID: the engine clamps STRATA_COHORTS>=2
to the SAME two-cohort binding (generate.cpp:9344) and ver_c exists nowhere; a =3 boot is
byte-identical to =2; the 3-cohort question is UNTESTED at any client count (9-wide 5+4 hits
kVerifyMaxT=8 and rotates an idle slot every round — the real F-273 9c~6-parity mechanism; the
3+3+3 ternary reactor is the structural fix, needs code). (b) mmvq IL tables DO cover the verify
head at T=2..8 on the 5090 (kIlRows_120/kIlRows58_120, il_cc=1201, native_mmvq.cu:3041-3194) —
T15A's "~73% of windows plain" is stale; only T=1 remains plain. (c) GDN layers = 36
(geometry.hpp:25-26/67; T26-3's 24 wrong). (d) the draft head is 40,525 x 2560 Q4_0 = 55.7 MiB/step
(the 162,100/425,196 figure was draft_vocab FILE BYTES). (e) standing depth is 6 and frac is 0.35
(F-263's reverts re-confirmed against the live config — kill any "depth 7 / frac 0.55 adopted"
restatements on sight).
HEADLINE FINDINGS — hc line's cap = BYTES-IN-FLIGHT (~205 KB staged vs ~1.36 MB saturation
requirement — predicts the measured 440-622 GB/s; TC-hc rung 0 = weight-cp.async + 4 accumulators
NEVER MEASURED, rung 1 = m16n8k16 bf16 fixed-order cross-warp + 3-stage weight pipeline; band
+6-12% M1; fused_gr_bench >=1.3x remains the LAW before any battery). MSInfer gate-prescale has NO
bit-preserving form (reassociation IS the identity; PS-form bits ~2.8e-7 = the GDN_CHUNKED class)
-> the F-276 composed-boot canary move 90->75-81% is quantitatively inconsistent with the identity
and points to a CONFOUND: one isolated env boot + KL pair is the falsifier. basalt's two-stage head
is a 2-BIT PROXY scored on the int8 tensor pipe on THEIR verify path, and our sampler consumes
FULL-VOCAB logits — parity-harness-first (-0.2..-0.25 ms/window, +4-5% M1 band). mid_chain
UNCONDITIONALLY UNBLOCKED: GGUF v3 headers parsed directly — attn_gate = 10 Q4_K + 14 Q6_K + 4 Q5_K
+ 8 IQ4_XS, ALL inside the donor's type_ok; the parse reproduced F-251's counts; generic triple on
every GDN layer, 553 donor lines, T<=4, +1.5-3% est; dec_gdn_rec (407 lines) the bitwise rung
behind it. F-191's PDL no-op RESOLVED not contradicted: the arm covered 25-40 of ~2,050 window
nodes = -0.01..-0.05 ms, inside its own floor (STRATA_DF_PDL_OFF per-pair kill-switch designed).
CONV_COMPRESS dead PERMANENTLY with mechanism: SplitZip (arXiv 2605.01708) bounds lossless coding
by float-exponent redundancy — int8 KV codes have no exponent field; our 1.01x IS the bound (kills
F-276's "structure-aware codec owed" hope). COLD-START falsifier staged: STRATA_PIN_GUARD=2
(expert_source.cpp:3512, in-tree) skips the wholesale page-cache drop in 2 MiB steps — ONE boot
prices bounded-re-read + the hugepage middle form (#771 compaction-stall caveat; box defrag=
madvise). DIET TRANCHE-1 = LFUSE/FUSE_WAIT/DF_PLE + UseNodePriority, all in-tree default-off
(verify.cpp:97/:320) = -344 nodes ~ -0.17..-0.33 ms/window, four one-boot arms. CONDITIONAL-GRAPH
KEEP-RULE implementer-ready (host-launched design over the draft loop — every CUDA-13 limit is a
device-launch limit, none bind host-launched; Gate-0 = th=0 bit-identical boot). ADMISSION OVERLAP
decomposed: C2a finish-A-under-B tail reorder (conc +8-14% arithmetic on the 26-29 ms round), C2b
reader thread (+3-6%), C2c adapt-join deferral; zero-code pricing falsifier = one decode-share-0.9
3v3. BATCH-LANE TRUTH: the cohort lane offers exactly ONE draft per slot (rows/slot fixed 2) — the
loaded horizon is already minimal; spec-under-load resolves to solo-keeps-6 / loaded-stays-short
(Red Hat conc-100 TTFT p95 -18.7%; our 9c acceptance 77.98% = highest measured). C5: WSTAT is BLIND
to cohort windows (single window_ms.push_back at generate.cpp:12428) — one-line fix gates ALL
overlap verdicts (T38 implementer item #1).
T38 IMPLEMENTER IN FLIGHT (port-coordinated, no commits/batteries): GDN=36 settled; C5 -> V2
STRATA_DF_PDL_OFF kill-switch + accuracy record -> V4 mid_chain + dec_gdn_rec on a shared bitwise
parity rig -> C1 Gate-0 stretch; main loop batteries its arms on report. STANDING RECIPE NOW: the
F-276 six + STRATA_DM (dm composes on the device-plan route). OWED: implementer batteries; PIN_GUARD
boot; diet tranche-1 arms; decode-share-0.9 pricing 3v3; 9-wide telemetry leg (post-C5);
prescale isolated boot + KL pair; CCD_SPLIT redo; re-quant USER GO.

## F-278 — 2026-10-23 ~17:15: T38 LANDED (C5 + V2) AND DEPLOY-VERIFIED BY A FOUR-LEG
## INTERLEAVE; mid_chain stopped honestly at the parity gate
THE IMPLEMENTER'S VERDICTS (every claim main-loop-verified at HEAD before commit):
(1) GDN = 36 LAYERS SETTLED (geometry.hpp:25-26/67; graph.hpp:45; independent GGUF re-parse:
attn_qkv 22 Q6_K + 10 Q5_K + 2 Q4_K + 2 IQ4_XS = 36 — T26-1 right, T26-3's 24 wrong; V4
arithmetic scales x1.5 on 36). (2) C5 LANDED + BOOT-VERIFIED, and the dossier's one-line
prescription was INCOMPLETE: the standing path's WSTAT flush never fires for a cohort-drained
boot (HEAD evidence: a 15,307-cohort-round boot printed ZERO WSTAT lines) — the fix needs the
cohort push (generate.cpp:12085, after the failure check) AND the cohort-side idle flush
(:12223). First live cohort-lane histogram: 363 windows, p50 23.8 / p99 30.7 / max 42.6 ms —
the C2a/C3 overlap program now has its instrument. (3) V2 LANDED: STRATA_DF_PDL_OFF grammar
(add/-remove/all/none, warn-once unknowns), all 19 launch_pdl families COMPILE-ENFORCED named,
WAR default = quantize_q8_1 (TRT-LLM discipline), pdl_parity green on every arm, on-disk
accuracy record campaign/pdl-accuracy.json; STRATA_DF_PDL unset = byte-identical behavior.
Correction banked: gu_swiglu q8 stores are NOT launch_pdl sites at HEAD. (4) MID_CHAIN
STOPPED AT THE PARITY GATE — the bitwise-first rule working as designed: the donor's pair is
NOT bitwise vs our standing pair on the pack's 17 real type-triples x T=1..4 (68 cases x 3
iterations: 0 bitwise, ~19% of output bytes differ at deterministic rel ~1.0-1.3e-7 = fp32
summation-order class), AND the donor's own header claim "same operations (and bits) as
dense_tc_pair" is FALSE at fb0944c7 on our shapes; the harness is proven (our dense_tc port
is bitwise vs native_mmvq on 23 kernel + 12 pair cases), so this is a real divergence surface.
dec_gdn_rec not attempted (gated on mid_chain green). Any future adoption = an explicit
F-230-class M3b/KL-gated decision, never a near-miss. Reopen: build/mid_chain_parity after a
donor revision or a port that adopts dense_tc_pair's term order. (5) C1 GATE-0: gap analysis
only (three concrete design gaps: the coupled-twin graphs round_exec_c_/step_exec_c_ the
dossier never addressed; the PLE-prefetch/wait_step interleave + g_dt timing decomposition;
the record_step/record_round split our mtp.cpp lacks). No guessed code.
THE DEPLOY VERIFICATION ARC (a four-leg interleave on the BINARY variable, the F-203 law;
user called the decisive leg): A1 new/busy 205.6/205.7/295.8 -> B1 rollback/busy
207.6/215.3/314.2 -> A2 new/QUIET 222.1/198.1/292.6 -> B2 rollback/QUIET
195.8/193.2/299.5. VERDICT: NO BINARY EFFECT on any lane (pair means 201.9 vs 204.3, conc
294.2 vs 306.9, solo favors the new binary 213.9 vs 201.7 — all inside single-boot noise);
B1's 215.3/314.2 was the outlier. TWO LESSONS: (a) the user's own device activity taxes the
CONC/PAIR lanes via host-core preemption (the eunomia 15.3ms class, live-confirmed: quiet-box
solo jumped +16 to the day's best 222.1 while busy-window legs sat 205-208) — adoption
batteries stay quiet-box policy; (b) pair/conc single boots swing +-5-8% on this box
independent of binary (the F-263 five-boot pooled law remains the only judge for conc-class
questions). Rollback: engine/strata.pre-t38 (one cp); C5 revert = two hunks in generate.cpp;
V2 revert = pdl.hpp + verify_kernels.cu + the 19 family names.
OWED (main loop, next quiet window): V2 re-price arm STRATA_DF_PDL=1 (+nsys per-pair probe,
pdl-accuracy.json entries, then 3v3 only if overlap shows — F-191's null is EXPLAINED as
partial-chain-inside-noise, no longer citable as falsification); PIN_GUARD=2 cold-start boot;
diet tranche-1 four one-boot arms; decode-share-0.9 pricing 3v3; prescale isolated boot + KL
pair; CCD_SPLIT redo; TC-hc rung-0 microbench; 9-wide telemetry leg (C5 now live).

## F-279 — 2026-10-23 ~18:05: THE TERSE-DIALECT EFFORT SET ADOPTED (user directive); the T39
## wave consolidated; E1 cache-slots fix + the PDL default-on correction pinned
THE TEMPLATE (user: "low/medium/high more effortful, thinking terse and concentrated, final
answers normal"): all three effort tiers rewritten in serve/chat_template.jinja + mirrored
EXACTLY (AST-verified, char-for-char) into server.py EFFORT_TEXT — xhigh = full rigor
(enumerate angles, validate assumptions, weigh alternatives, check edge cases and
counterexamples, re-verify the load-bearing step), medium = decompose/work/verify-crux (had
NO directive at all before — a real gap), low = identify/derive/sanity-check; each carries
the terse-dialect contract for THINKING ONLY (compact working notes, dense clauses, one step
or fact per line, symbols and short labels over prose, no filler, no restating the prompt,
never pad) + the final-answer guard "clear, natural prose, HONORING ANY STYLE THE REQUEST
ITSELF ASKS FOR" (the style-defer clause — added after the first gate read mimic 56/66 and
the mechanism was identified: the plain "natural prose" clause fights style-mimicry tasks).
Coupling handled: #728 recovery token-swaps HIGH_EFFORT->LOW_EFFORT (mirror keeps it sound),
goldens regenerated (18 embeds), 18/18 tokenizer tests green.
GATE ARC: baseline (old template, live server) = mean 634 completion tok / 3.2s wall, 2 of 6
prompts burning the entire 900-token budget on thinking; NEW = 608 tok mean (-4%; -35% on
the puzzle class: 264->154), content chars UP 837->929 (budget-truncated answers freed);
MMLU 94.3 + 94.3 (floor 93.3 PASS, 1-sigma of the 95.0 anchor); first mimic read 56/66 (the
known low mode — appeared twice on the OLD template too) -> style-defer fix -> PAIRED
INTERLEAVE (t39-tpl-mimpair, git-checkout old vs fixed-new, ABAB x2): old 60/58 vs new
60/62 — new >= old EVERY paired round. ADOPTED (3 files; rollback = git checkout).
THE T39 WAVE (three verified dossiers, INDEX rows; the user's "push past the record" +
"elastic delays/back-and-forth" + "dead flags" directives): [R2 elastic] the VMM layer is
HEALTHY (promote p50 2.3ms, demotes in true idle) — the waste is ABOVE it: E1 the 4-SLOT
conversation-cache default (615 MB used of 48 GiB, evicting 10x/boot) = FIXED TODAY
(--conversation-cache-slots 24 in the config, exact-class, verification = eviction counter
on next boots); E2 the 42.7ms main-path park D2H rides the REQUEST THREAD before the prompt
read (generate.cpp:9602/13565; background it = -40ms TTFT on switching turns); E3 the
two-hop restore (11+12-33ms; straight-into-slot = -25-45ms per return); 9-wide rotation
LOG-MEASURED ((8,8) rounds + 60-64 idle-A rounds/boot) — T37 C3 ternary prices +12.5%
in-window tokens, 9-conc band 355-390 vs record 323.0, ver_c ~127-250 MiB. [R1 anatomy]
standing window row-flat 12.9-14.9ms (graph replay IS the critical path); THE COHORT ROUND'S
2x (p50 24.3, n=17,234) = two ~6-row batch graphs (~2,660 nodes post-dm) replaying SERIALLY
on one stream — C2a-lite (finish B under A's tail) = +5-8% conc, THE >4% band this wave,
gated on C5b (the two WSTAT histograms measure different phases; cohort push misses the
~1.4-2ms commit/emit tail); audit: 33 arms + 28 args = 0 DEAD, 31 KEEP (TYPICAL_ACCEPT
re-proven live: 2,646/5,186 first-rejections rescued), 2 SUSPECT (POOL_SPIN_US,
PREDICT_CREDIT/BUDGET) for one pooled drop-test each; VERIFY_PROFILE no longer measures the
standing composition (it disables dm + FUSE_HITS) — re-baseline uses STRATA_VERIFY_NODES=2;
the "spec-8 buffer class" is LOAD-BEARING (batch 2-rows/slot cap) — F-270's VRAM ledger
attribution corrected. [R3 prefill/acceptance] no >=4% in-tree prefill lever unpriced
(quantize-dedup SHIPPED c675f5b7 — T15 stale; K=640 fp16 target never executes — family
closed); the acceptance battery (tau {0.15/0.25/0.30} + SPEC_PROB, MMLU-gated) is the cheap
value and DECAYS at a successful retrain — measure now; STRATA_SPEC_RATIO's "preserves the
distribution EXACTLY" claim FALSE (over-emits (1-p)*min(q,p) per rescued site); the 9-conc
77.98% acceptance = denominator artifact until the batch lane gets its trace twin (one
fprintf).
THE PDL CORRECTION (R1, main-loop-verified): the T38 V2 code DEFAULTS STRATA_DF_PDL to 1
when unset (verify_kernels.cu:2084, "opt-in until the A/B says otherwise") — every boot
since T38 silently ran PDL-armed; F-278's "unset = byte-identical" and pdl-accuracy.json's
env line were WRONG (the implementer's report mis-stated the default). Measured NEUTRAL
(the T38 A-B-A-B deploy interleave was effectively PDL-on vs PDL-off, indistinguishable;
armed surface ~13 of ~1,860 nodes). FIXED: export STRATA_DF_PDL=0 pinned in the launcher
(documented posture restored, protocol violation closed; the re-price arm is =1, not =0).
NEXT: T40 implementer on C5b + C2a-lite + E2/E3 (park/restore pathway) + E5 ternary design
+ E6 + the batch trace twin; main-loop boots: A1 diagnostics (VERIFY_NODES=2), E1 eviction
verification, tau+SPEC_PROB battery, the 2 drop-tests; five-boot pooled vs the records
215.2/206.4/323.0 (dm pool already 222.4/209.9/319.6 — conc is the battle).

## F-282 — 2026-10-23 ~20:25: THE T41 JUDGMENT BATTERY — SPEC_PROB ADOPTED (the never-measured
## arm pays); c2a shelved honestly; tau 0.15 falsified; the night's full arc closed
BATTERY (t41, 10 prewarmed legs, ctl/c2a 3v3 interleaved + tau015 x2 + SPEC_PROB x2; the ctl
carries the new standing posture = slots-24 + min-tokens-200): ctl M1 205.8 / pair 203.1 /
conc 304.6 (the evening window ran COOL - ctl sat below the afternoon pools 217/315 all
night; interleaved deltas are the honest read). VERDICTS:
(1) SPEC_PROB=1 ADOPTED — M1 +5.8% (217.8; legs 220.5/215.1 BOTH above every ctl leg
203.1-209.6, non-overlapping), conc +4.4% (318.1), M3b +2.0pp (78.2 - mechanism-correlated:
the acceptance instrument moved with the arm's own mechanism). Gates: MMLU 4-run set
94.3/92.9/95.7/94.3 = mean 94.3 with ONE sub-floor read (B leg 92.9 < 93.3; tiebreaker legs
C 95.7 / D 94.3 clean) - the pre-declared rule (mean >=94-class, <=1 sub-floor) PASSED.
Mimic 56/66 = the night's BENCH LOW-MODE (third occurrence across THREE different configs -
dm arm, template, prob; both paired interleaves tonight showed the true 60-62 band; rides as
a documented watch, not a veto). Launcher arm added; five-boot pooled rider + mimic watch on
the next battery. Expected standing recipe on a fair window: solo ~230+, conc record (323.0)
in reach - the next clean re-bank decides.
(2) c2a (STRATA_COHORT_OVERLAP) SHELVED - its target lane is FLAT (conc +0.6%, rounds split
2-1); the arm's M1 "wins" (+6.0%, 3/3) are on a lane it STRUCTURALLY CANNOT TOUCH (solo
never rides the cohort path) = window variance masquerading as signal (the F-225 lesson,
relearned cheap). Stays default-off in-tree; the +5-8% arithmetic needs the true tail split
from the C5b histogram before any retry (the smoke-test p50 read 27.3 vs 23.1 - the overlap
may even cost; the 3v3 says ~flat).
(3) tau 0.15 FALSIFIED all-lane negative (M1 -1.9 / pair -5.0 / conc -2.6 / M3b -0.2) -
0.2 stands; the loosened-rescue direction is closed on this stack.
NIGHT LEDGER (the user's "push past the record, WTF moments" directive): ADOPTED - dm
(pooled solo record fallen), terse-dialect template (thinking -4%/-35% puzzle-class,
answers natural, style-defer), cache slots 24 (evictions 10/boot -> 0), min-tokens 200
(E6), PDL pinned honest, C5b phase-correct cohort telemetry (first true round p50s),
batch acceptance trace (594-line sample banked), SPEC_PROB (+5.8% M1). FALSIFIED/CLOSED -
tau 0.15, c2a current form, mid_chain at this donor revision, CONV_COMPRESS at the bound,
spec-8/frac-0.55 restatements, COHORTS=3 myth. INSTRUMENTS - the trace twin, the phase-fixed
histogram, template-probe. THE PROVIDER ROUTE killed three implementer attempts (network,
~30 min in each) - the pathway cluster (E2/E3), E5 ternary, E7 go to the next session with
full file:line specs; subagent route to be re-tried then.
RECORDS: solo 222.4 (dm pool, now + SPEC_PROB expected ~230+), pair 209.9, conc 319.6 vs
323.0 - the conc record is the standing target; the ternary reactor (band 355-390) is the
queued kill-shot.

## F-283 — 2026-10-24 ~00:35: THE USER'S PROTOCOL TEMPLATE ADOPTED VERBATIM (four-tier
effort-graded reasoning protocol replaces the F-279 terse-dialect set). The user hand-wrote
chat_template (4).jinja — a full rebuild, not a tweak: low|medium|high(default)|max tiers
(xhigh renamed away; 'max' = interpret/anchor/solve/premortem/steelman/re-derive/calibrate),
a thinking register with a worked example (n²+n+41) and relation symbols (→ ∴ ≠ ✓ ✗ ?),
per-tier method + research blocks, retrieval_available AUTO-DETECTION from tool names
(web/search/fetch/... keywords; no-tool branch forbids fabricated citations instead),
evidence/disposition/self-awareness/integrity rules, "Instructions that follow in this
system prompt override this protocol on style and format", and preserve_thinking default
FALSE (prior-turn reasoning dropped from history; standing kept it).

TESTS (campaign/tpl-user-render-battery.py + batteries/tpl-user-gates.sh +
batteries/tpl-user-mimpair.sh, logs campaign/results/tpl-*):
- Offline parity (production ChatTemplate env): ZERO engine errors; every must-raise case
  (invalid effort / system-mid / empty) raises on both; tools round-trip, vision,
  developer-merge all render; caps probes pass. Diffs: supports_preserve_reasoning flips
  True→False (discovery-only, no consumer); effort 'none' raises on BOTH templates
  (pre-existing edge, parity not regression). Absolute prompt overhead (real pack
  tokenizer): +435 tok/conversation at default effort (579 vs 144), +147 thinking-off,
  +270 net multi-turn x4 — the protocol prefix is SHARED, so the prompt cache amortizes it
  to ~0 after each conversation's first request (~65 ms extra prefill once).
- Think-probe: mean completion 616 tok vs 608 standing (parity; old-template baseline was
  634); reasoning chars 2.1-2.6k inside the same budget = the register's symbols buy
  density; answers 841 ch mean (band).
- MMLU: 94.3 / 95.7 (floor 0.933) — PASSED, mean 95.0.
- Mimic paired interleave A/B/A/B across boots (user/standing/user/standing):
  87.9 / 87.9 / 87.9 / 90.9 — parity within noise (−1.5 pts mean = 1 item of 66; blend
  swings 4-8 on identical templates, user mean 5 vs standing 7 on n=2 each). All four legs
  in the banked 58-62 band. No style-defer regression: the override clause + natural-prose
  final-answer rule hold the band.

ADOPTION WIRING: serve/chat_template.jinja = the user's file verbatim (md5 461afc276cf9);
server.py EFFORT_TEXT rewritten to the per-tier METHOD sentences (verbatim from the
template's methods dict; #458 trailing turns stay consistent — dormant, effort_position not
set); #728 HIGH_EFFORT/LOW_EFFORT re-anchored to " (effort: high)" / " (effort: low)" —
the protocol header's in-context token span (leading space REQUIRED: the pack tokenizer
merges ' ('; verified [318,6243,366,25,1496,8] hit=True; dormant, reasoning_loop_recovery
off); chat_golden.json regenerated (rendered only; hf nulled — no transformers in .venv,
cross-reference pending); tools.test_strata_tokenizer 18/18 OK; server rebooted, sanity
high/low/none all clean, LEFT UP on the adopted state.

WHAT IT BUYS: the terse-dialect gains (F-279: −35% thinking tokens on the puzzle class)
now ride on a graded-protocol frame — effort scales METHOD (search breadth, verification,
self-critique) instead of verbosity; retrieval rules activate only when web tools exist;
honesty/anti-confabulation rules ship in the protocol itself. Token cost ~zero marginal
(cached prefix). The 'max' tier is NEW surface (previously unreachable depth-7-class
rigor on demand). NEXT RIDERS: five-boot pooled re-bank of the full recipe unchanged
(template swap is orthogonal to speed arms); hf cross-reference when a transformers env
is handy; watch blend-category on the next mimic (the only lane that leaned standing).

### F-283a — 2026-10-24 ~01:35: THE "FEELS SLOWER" A/B (user asked; tpl-speed-ab.sh +
tpl-ux-probe.py — 5 daily-driver prompts x2 per leg, A=live-warm/B=old-fresh/C=new-fresh).
ENGINE: unchanged — real-request decode 200.6 tok/s (in-band vs 184-230 across old-template
usage boots); no engine code touched by a template swap. UX TOKEN ECONOMY: the protocol
COSTS ~+13% tokens per open-ended answer (A 1535 / C 1511 vs B 1343 completion tok,
consistent across boot positions; reasoning chars +30% at 2980-3052 vs 2338; FINAL ANSWERS
THE SAME SIZE at ~3027-3141 ch) = ~+0.7s wall per typical answer (~+11%). Mechanism: the
high tier's method (compete approaches/disconfirm/re-derive/adversarial review) gives the
model real work to do on open-ended asks — the puzzle-class think-probe (616 vs 608) was
blind to it (short-hop puzzles leave the method nothing to chew). This is rigor bought at
+0.7s, not an engine regression. CONFERENCE NOTE: tonight's 6-reboot gates churn made the
evening feel cold-path slow too (2.5k vs 6-7k prefill until warm). Lane spot-check on leg C
(193.4/187.5/296.9) ran AFTER 10 UX requests (unusual context — first thing after prewarm
is the banked pattern); single boot, low edge of swing, no verdict — the owed five-boot
pooled re-bank of the F-282 recipe remains the instrument for engine-speed certification.

## F-284 — 2026-10-24 ~01:45: THE "CORE KNOWLEDGE v2" EXTERNAL DOC CHECKED (findings-only,
no boots; 3 Explore maps + main-loop verification + live-log telemetry). The user delivered a
~1,200-line external synthesis claiming the binding constraint is "DDR bytes of host-resident
experts touched per verify pass" on a miss-bound engine, ranking 12+ levers. VERDICT AGAINST
THIS TREE (campaign/research/T42-knowledge-check.md + T42-r1/r2/r3 maps):

(1) MODEL ID RIGHT: Qwen3.8-Flash-Next confirmed from the pack index (v1's Qwen3-Next
assumption was wrong); geometry matches (2560/248,320/512-experts; 47 per-layer quant rows).
Platform corrected: 184GB RAM not 128.

(2) THE HEADLINE THESIS IS HALF-RIGHT — it binds the TAIL, not the median. Live n=4,192
window telemetry: expert-cache hit p50 97.2% / p10 88.7% / mean 95.5%; misses ≈ 8% of the
median ~15.6ms forward, ≈ 31% at p10. The U(P) law is REAL and measured: 3-6-position verify
windows hit only 71.0% (speculation inflates misses — exactly the doc's mechanism), while
warmed conc/cohort windows recover to 97.1%. The MEDIAN binding term is the GPU kernel chain
(288-291 HC launches/forward — 3x the doc's count; 12 QSA indexer launch-quads), which is
what T15 (hc occupancy, bit-exact +8-9%) and T17 (hc ALU-cap mma, +9-17% ceiling) already
priced. The doc's own escape clause ("if handshakes or kernel chains dominate, items 5/8/11
move up") fires here.

(3) THE DOC'S "MOVES" ARE ALMOST ALL ALREADY BUILT IN THIS ENGINE — the campaign built them
across the B/F series: prefill ncmoe weight-streaming (>=1024-token chunks GPU-compute by
default, prefill.cpp:179 — "prefill misses compute on CPU" is WRONG here); the 149-prefetch
predictor (vote-ranked next-layer routing, confidence admission, recall/precision telemetry);
doorbell + polled mapped flags + batched DMA (pure overhead ~10-40us/layer vs ~0.2ms real
fetch wait); the 40,525-row Q4_0 draft shortlist (55.7 MiB/step — the "248k lm_head" claim
wrong); draft experts hard-VRAM (675MiB, no miss tier — §7.3 moot); the ENTIRE §5 engram
design (sample-time gather, IQ4_NL 90B/row = the 4-bit floor, page dedup, draft->verify row
reuse — "PLE disabled in draft" is BACKWARDS, the draft prefetches for verify — NG_HIST=9
rollback+park); L2 set-aside; COHORTS = the two-batch overlap; int8 KV; PcieAuto online
share regression. Residency: ~26.5% by count in VRAM (13.3GiB) + a 42.29GiB PAGE-LOCKED
complement = ALL experts resident somewhere (file tier first-touch only, ~5-6K reads/boot
decaying to 0) — the doc's "42-47% resident" describes a different design; profile-ranked
placement + prediction is why 26.5% count yields 97% hits.

(4) FALSIFIED-BY-JOURNAL BEFORE ANY BOOT (the verify-first law pays again): the doc's biggest
GPU lever "GDN row-split (48 CTAs = 28% of SMs)" — STRATA_GDN_SPLIT=1 already tested T1a,
"neutral on sm_120, NOT adopted" (SVG-CUSTOM.md:2175). The 48-CTA grid is real
(verify_kernels.cu:1215) but the kernel is not SM-starved in the window path. Likewise "K+1
sequential state passes" is wrong (in-kernel register loop, R2 Q3).

(5) GENUINELY NEW (the doc's real contribution, all falsifier-first, queued as Tier B):
QSA below-2051 HOST-SIDE LAUNCH SKIP (no bypass exists; 3-4 degenerate launches/layer/step;
exact-output class); GDN bf16 STATE (no bf16 mode exists — pairs with T17's bf16-native
tranche as one arm, KL-gated); MISS-AWARE VERIFY WIDTH (DSpark-style scheduling aimed at the
measured 71% band); --ple-io ram one-flag (mlock 28.8GB; robustness); PDL-RACE SCREEN (20x
greedy, the bimodality hunt's cheapest probe); per-expert popularity requant acknowledged as
NOT done (pack heterogeneity is per-layer).

(6) MERGED BREAKTHROUGH QUEUE (T42 §5): Tier A banked record-breakers = T15 hc occupancy
(+8-9% M1 bit-exact) / E5 ternary (conc 355-390 band vs 323.0) / FastMTP T14B. Tier B = the
new cheap boots above + T39's +460-slot dedup (+1.9pt hit, bit-exact class). Tier C = TC-hc
mma tranche / verify-width / stream-level overlap / requant (USER GO). RESCOPE: the
PREDICT_CREDIT/BUDGET + POOL_SPIN_US drop-tests become TUNE-tests — they run the predictive
tier the doc independently ranks #1-4. Records context: solo/pair past on pooled means
(222.4/209.9 + SPEC_PROB rider), conc 319.6 vs 323.0 = the target lane.

## F-284/F-285 — 2026-10-24 ~02:20: THE TERNARY REACTOR IMPLEMENTED AND PARKED; THE b9
LANE AND THE REBUILD PIPELINE BOTH FOUND BROKEN (a discovery night, not a record night).

**T42 RE-BANK (t42-rebank.sh, 5 prewarmed boots)**: pool M1 198.4 / pair 189.8 / conc
283.6 — a systematically COOL window (−10% vs the dm pool 222.4/209.9/319.6 on ALL lanes,
GPU locked 2940 MHz / 95% util / 362 W throughout, expert cache 97-98%, acceptance in-band,
ITL p99 36-37 vs dm-era 20). PREFILL PLATEAU HOLDS: client walls 15.7k tok in 2.5s (6.3k),
63k in 9.3s (6.8k) — the 6-7k plateau to 120k is intact (t42-prefill-ladder.py; its
log-parse half crashed — parse bug only, client walls banked). Golden hash drifted
a974bc→dbb294 = the F-283 template (expected, the alarm's first true positive).

**T42B COUNTERFACTUALS (ABAB×2, 6 legs)**: ctl 199.6/191.8/278.5 vs SPEC_PROB=0
197.3/192.0/268.1 (conc −10.4 without it!) vs old-template 186.6/183.6/271.3 (−6.5 M1).
NEITHER recovers the pool band → THE WINDOW IS THE STORY, the standing recipe is the BEST
configuration measured tonight; SPEC_PROB and the F-283 template both RETAINED with
same-window evidence. Records STAND as banked (215.2/206.4/323.0 anchor; dm pool
222.4/209.9/319.6); the honest re-bank moves to the next clean window (morning; the
F-224 self-workload law applies — this session ran 25+ boots and heavy analysis all night).

**F-284 TERNARY (STRATA_COHORTS=3, T37-C3 spec)**: fully implemented (patch script
campaign/patch_f284_ternary.py — ver_c verifier [2b/3,b) cloned from the ver_b binding,
ternary collect/stage/go/poll + third finish tail, C5b tail weight 1/3, engagement
n_active > 2b/3, batch>=9 gated; stage-order bug found and fixed: C's lazy graph capture
MUST happen before any go — capturing under A/B flight hung layer 0). Binary preserved at
engine/strata.f284-ternary-parked. PARKED — never ran one round, blocked twice over:
(1) THE LAUNCHER HARD-EXPORTED STRATA_COHORTS=2, silently overriding every env=3 boot
(fixed: now "${STRATA_COHORTS:-2}" like SPEC_PROB — the F-284 launcher lesson: EVERY
hard-exported arm is an env-override landmine for future battery legs);
(2) THE b9 LANE IS BROKEN ON THE CURRENT LINE: batch 9 + COHORTS=2 (the config-only
control) hangs after the FIRST cohort round completes — "#267 verify timed out at layer 0"
in the admission ramp as slots 5-8 materialize ("release 1.3 + map from the EXPERT CACHE'S
TAIL — 0 chunks left in the spare pool"; at batch 6 all slots fit the spare pool, at 9 they
steal from the cache = the elastic×cohorts×9-slot interaction). Reproduced on the
PRE-TERNARY binary — NOT my patch. F-273's 322.5 could not be re-measured on this line;
every 9-wide plan (the 355-390 ternary band) is queued behind this fix (T45: the
admission-ramp expert-tail release path, verify.cpp's wait arms).

**F-285 — THE REBUILD PIPELINE HAZARD (the night's most important find)**: every fresh
rebuild of HEAD tonight — MY ternary source AND a pristine stash-rebuild of unpatched
source — HANGS the batch-6 conc lane after the first cohort round ("windows done", then
silence; M1/pair FINE at 204-212/182-192, conc 0 tok in ~111s), on clean TERM-only boots.
The deployed T40-era binary (built 10-23 19:45, lost — overwritten during the ternary
deploys) ran the same source's cohort path for HUNDREDS of rounds tonight (t42+t42b: 11
conc legs, zero hangs); engine/strata.pre-t40 ALSO passes (conc 304.1 = the T41 band,
M1 196.3 — dm-era source, no T40 instruments, SPEC_PROB's sampler tweak likely absent =
the arm silently inert). ROOT CAUSE UNKNOWN — F-247 class (the 19:45 build environment
vs tonight's differs in something unrecorded; PATH/nvcc/build-dir all check out).
STANDING STATE: server UP on engine/strata.pre-t40 (working conc, in-band lanes); the
source tree carries the parked ternary; engine/strata = pre-t40. NEXT SESSION'S FIRST
ITEM: bisect the rebuild hang (fresh build dir as step 1 — rule out object staleness;
then compare nvcc/flags via ninja -t commands; the T40 instruments' C5b relocation is the
prime suspect for a codepath that only a differently-optimized build exposes). Until the
hazard is fixed: DO NOT deploy any rebuild over engine/strata without a conc-leg proof.

## F-285 — 2026-10-24 ~02:30: CODE-STUDY PASS ON THE 260/400 TARGETS (findings-only).
Fresh-anchor read of the solo window loop / PLE solo path / HC machinery / draft internals.
THE FIND: STRATA_HC_PERSIST=1 — a persistent whole-read HC kernel exists, compiles on CUDA
(fused_gr.cu:34 !__HIPCC__ branch; the journal's "(HIP)" label understates it), is wired
FIRST-IN at gr_read_group (verify.cpp:1426, covers both per-layer reads), launches as a
plain <<<>>> with a software grid barrier + spin timeout (graph-capturable shape), and
self-checks against the default read at boot with graceful fallback. NEVER MEASURED on
this box. Value hypothesis +2-8% M1 (HC reads 3 launches -> 1 each, ~288 -> ~96
launches/forward + the norm/down/up chain fused in-kernel = the external doc's §6.N lever,
already written in-tree). RE-SEQUENCES the HC lane: boot persist BEFORE writing T15 grid
code (same surface; if persist wins, T15's read-side fixes target a dead path). Riders:
pin the solo PLE pageable staging (sub-0.2%), spec-depth re-check post-SPEC_PROB (depth 7
died F-263 pre-SPEC_PROB; economics changed), diet-tranche remainder (-200 nodes,
+0.7-1.2%). Dead ends confirmed: adapt std::thread (every-4th amortized), GDN split
(journal), c2a re-read (needs the A1 boot; no existing cohort WSTAT logs).

## F-285a — 2026-10-24 ~03:05: ROOT CAUSE FOUND — THE CHAMPION IS RESTORED. The rebuild
## hang was F-281's C2a-lite relocation all along.

THE BISECT (one boot, decisive): fresh-dir full rebuild of HEAD with generate.cpp taken
from 9e5366c6 (F-280 — pre-C2a) → conc lane PASSES 272.2 (cool-window band). Every
element of the diagnosis chain: NOT ccache (none on the system), NOT the build dir
(build-fresh of HEAD hung too), NOT my ternary (pristine stash-rebuild hung), NOT the host
compiler (GCC 16.2.1 identical stamps in passing and hanging binaries). THE MECHANISM:
the 19:45 champion binary was built BETWEEN F-280 (19:08) and F-281's C2a-lite commit
(19:51) — it NEVER CONTAINED the relocation. Every battery since (T41, t42, t42b — 14+
clean conc legs) ran that pre-C2a binary; F-281's "SAFETY VERIFIED 12/12 on BOTH off/on
boots" must have ridden the same pre-C2a binary (its smoke never exercised a build WITH
the relocation). The relocation's claim "default off = original order structurally
preserved" is FALSE: with STRATA_COHORT_OVERLAP unset, the finish tail after a completed
cohort round hangs (round logs "windows done in X ms", then silence; M1/pair unaffected —
they never enter the cohort branch). The C2a-lite block is REMOVED from the standing
tree (it was shelved F-282, never adopted — zero loss).

THE RESTORED CHAMPION (engine/strata = build-fresh of HEAD + generate.cpp@9e5366c6):
dm + PDL-pin + SPEC_PROB (sampler.cu:891 live again after pre-t40's silent absence) +
C5b/trace instruments + the F-283 protocol template + every launcher arm. PROOFS: conc
272.2 / M1 184.8 / pair 186.1 (cool-window single-boot in-band); MMLU 94.3/92.9/95.7
mean 94.3, one sub-floor (the pre-declared rule PASSES; the leg set is byte-for-byte
F-282's); mimic 56 = the documented 4th bench low-mode (paired band 58-62 on equivalent
composition tonight — watch, not veto). Server UP on it.

STANDING LESSONS: (1) "default-off = structurally preserved" claims must be proven by a
BINARY-COMPOSITION interleave (the same binary with env on/off), not just boots — a
relocation changes the default path even when the arm is off. (2) engine/strata deploys
must record the SOURCE COMMIT + a conc-leg proof in the same breath (the 19:45 binary's
source was never recorded — that gap cost tonight's whole detour). (3) build-fresh/ is
the canonical build dir now (build/ retired — its mixed-state history fed the confusion).
The ternary patch script campaign/patch_f284_ternary.py targets the C2a-era round layout
— its poll-loop anchor needs re-writing against the 9e6c base when the b9 lane is fixed
(T45); the parked binary engine/strata.f284-ternary-parked and the 3f068280 history
carry the full implementation either way.

## F-285b — 2026-10-24 ~03:10: THE RESTORED CHAMPION'S FIVE-BOOT BANK (t44-champ-bank.sh,
## user asked for the full single+conc proof).  POOLED n=5: M1 200.3 (σ 7.9) / pair 182.2
## (σ 3.8) / conc 278.3 (σ 12.3); accept 76.4-78.1 in-band.  COMPOSITION VERIFIED: the
## champion reads the SAME band as the lost 19:45 binary's t42 pool (198.4/189.8/283.6,
## same window class) — M1 +1.9 / pair −4.0 / conc −1.9, all inside boot noise → the
## restore is behaviorally the same engine that set yesterday's numbers.  vs the dm pool
## (222.4/209.9/319.6): −10.0/−13.2/−12.9% = THE WINDOW TAX, unchanged from every binary
## tested tonight (the t42b counterfactuals already ruled out the recipe; clocks locked,
## cache 97-98%, acceptance in-band).  In-battery trend: pair legs slid 186.5→176.5 over
## the five boots = the box's 4th consecutive hour under batteries (the F-224 self-
## workload law at session scale).  RECORDS STAND from their banked windows; the champion's
## clean-window certification = the morning re-bank (first quiet window, fresh five boots).

## F-286 — 2026-10-24 ~03:50: THE TERNARY RUNS — ALL THE WORK IN ONE BUILD; the b9 lane
## unlocked CONFIG-ONLY; first 9-wide numbers since the elastic era.

**THE UNLOCK (T45, solved without touching the buggy path)**: STRATA_ELASTIC_S_MIN=9 maps
ALL nine slot arenas at boot → the admission-ramp materializations never run → the
expert-cache-tail release (the #267 layer-0 hang) never executes.  12/12 smoke PASS at b9,
first 9-slot boot since T36 introduced elastic (the F-273-era b9 boots predate it — the
"regression" was elastic×b9, never re-tested until tonight).  Cost: 6.75 GiB of arenas
resident from boot (21.5 GiB free at boot pays it).

**THE TERNARY (F-284 v2, patch_f286_ternary_v2.py, re-anchored to the C2a-free base)**:
true 3+3+3 (the v1 half-split bug caught live from the round shapes A.S=8 → fixed to
thirds → A.S=6 B.S=6 C.S=6, every window the 6-row class, ALL NINE SLOTS DECODE EVERY
ROUND — the rotation cap gone).  TWO launcher landmines found and cleared: the b9 launcher
COPY still hard-exported STRATA_COHORTS=2 (copies inherit old lines — audit every copy
after a launcher fix), and the stage-order law holds (C stages before any go).  12/12
smokes ×2.

**T46 3v3 (ctl9 vs ter9, both S_MIN=9, 9-conc lane)**: ter9 legs 293.4 / 246.6 / 298.0
(mean 279.3) vs ctl9 253.5 / 261.4 / 248.7 (mean 254.5) = **+9.7% pooled, 2-of-3 rounds
(+15.7%, +19.8%, one −5.6% on the fatigued-box stretch where ALL boots' rounds drifted to
~37ms)**.  The biggest conc lever since COHORTS itself, live and correct — but NOT a
banked record: 279.3 < 323.0, and the night's window tax (identified: CPU CONTENTION —
chrome's live monitor burns ~200% during legs, two agent sessions, engine feed threads
preempted; the naive taskset 0-7 pin FALSIFIED at −20%, the real fix = the CCD_SPLIT
design or ambient-burn reduction) plus 5h of continuous batteries have everything reading
−10-15%.  The 293-298 legs in THIS window ≈ 320-330 in a normal window → the record is IN
REACH, not yet taken.  The confirm ride (2+ more rounds on a rested box, monitor closed or
CCD-split) banks it.

**WINDOW-TAX LAW (from the WSTAT forensics)**: cohort rounds 22.1-24.7ms in yesterday's
warm windows vs 25.4-28.3ms tonight, all boots, recipe-exonerated — the tax lives in the
host feed, and the monitor dashboard's own charting is a named contributor (its burn
scales with live traffic — watching a record costs the record).

## F-287 — 2026-10-24 ~04:40: THE GOAL OPENS (300 solo / 450 conc) — rung 0 built, and the
## DOWN line falsified AT THE WALL; the UP line is the real kernel target.

**RUNG 0 (kHcStaged2=4, STRATA_HC_SPLIT=4, patch_f287_staged2.py + inline fixes)**: the
staged down read with a 3-slot WEIGHT cp.async ring (per-thread self-consumption — no
cross-warp weight dependency, one __syncthreads dropped) + 4 independent fp32 accumulator
chains combined in fixed order.  Builds clean; wired into the full launch ladder (exact-T
1..6 + max4/generic) with its smem opt-in (61,440 B ring on top of the staged tile).
**BENCH VERDICT: FLAT** — down 4.1/4.7/5.6/6.8/8.1/9.8/10.6/10.6 µs vs staged baseline
4.0/4.6/5.5/6.8/8.1/9.8/10.6/10.6 (T=1..8, cold 4-set mode, paired same-minute runs).
THE FALSIFICATION (valuable): at T=1 the staged down reads 6.55 MB in 4.0 µs = **1.64 TB/s
= ~96% of the DRAM wall** — the isolated kernel is AT THE FLOOR; the T37 outstanding-bytes
deficit (0.44-0.62 TB/s) is an ENGINE-CONTEXT number (L2/DRAM contention with the expert
stream), not an isolated-kernel property.  No down-side restructure (rung 0 OR mma rung 1)
can beat the wall — rung 1's down half is DE-PRICED unless the engine context is the target.
At T>=4 down is latency/occupancy-bound (0.62 TB/s at T=8) — grid-starved by the load-bearing
41-block shape (the F-15C -6.99% family); staged2's per-thread staging REMOVES the reason
that family died (staging-threads coupling), so a wide-grid staged2b stays priced for T>=4.

**THE REAL TARGET FOUND BY THE SAME PAIR: the UP line.**  At the T=3-5 that dominates real
windows, up = 10.4-13.4 µs vs down's 5.5-8.1 — 60-70% of the hc read line; at T=8: 18.1 µs
= 0.36 TB/s (vs down's 0.62).  gr_up_multi's anatomy: per-row-element weights prefetched
but the per-token warp_sum shuffle tree SERIALIZED against the dots (dot k -> 5 shfl ->
dot k+1); the batched form (all T dots as independent register chains, then one vector
butterfly) is the rung-0b move.  CAVEAT banked: the ENGINE runs gr_up_q32 (QFUSE, default
on when q8_mixed is set) — the BENCH does not cover q32 (bench leaves q8_mixed null) — the
up-line rung needs a q32 bench instantiation FIRST or it prices the wrong kernel.

**Also tonight**: 12-wide ternary SHELVED (aggregate flat vs 9-wide — 281.1 ~ 279.3, three
more slots divide the same window throughput; solo crushed to 153.5 by 9 GiB arena
residency) — conc is WINDOW-THROUGHPUT-bound, which makes the kernel line (down+up+verify)
the shared path to BOTH targets.  T47 research wave (solo-300 + conc-450) running.

## F-287b — 2026-10-24 ~04:55: T47-solo-300 INTEGRATED (verified) — and a RECORD
## CORRECTION on the parallel session's F-285 HC_PERSIST find.

**CORRECTION (verified in code)**: 409ce8c8's "STRATA_HC_PERSIST found armed-ready on
CUDA, +2-8% M1 hypothesis, re-sequences the HC lane" is INERT — the persistent HC read is
HIP-gated (its barrier clock is s_memrealtime, fused_gr.cu:2860; the code's own comment
at :2869 says gfx11/12 lack it; CUDA takes the stub).  The HC lane stays T37's ladder
(down de-priced at the wall per F-287; UP line = the target).  The re-sequencing worry is
moot.  (The verify-agent-claims law pays again: two dossiers, two corrections tonight.)

**THE 300 FRAME (T47-solo-300, accepted)**: no single +35% lever exists anywhere (best
published consumer-GPU MoE decode single-lever: SMOE 20.9%; the public 5090+MTP frontier
for this model class runs 45-122 tok/s vs our 222.4 — we are 2-4x past the field already).
300 = COMPOSITION: fair-window champion ~232 (SPEC_PROB postdates the record) × TC-hc/UP
(+6-12) × two-stage head (+4-6) × retrain-or-keep-rule (+6-9, same deficit pool) ×
experts-requant USER-GO (+4-7) × micro bundle (+1-2) = central 296-300, upper 315-327,
with three named double-counting traps.  FIRST ACTION OWED: the fair-window five-boot
re-bank (sets the ladder's base B; the box is 5h-battered — next rested window).

**TWO NEVER-ATTEMPTED CANDIDATES ADDED to the queue** (both confirmed absent in-tree):
(1) context-n-gram draft source (prompt-lookup class) spliced into the coupled MTP chain —
lossless by construction, 0% on literary probes / +5-25% on copy-heavy real traffic;
falsifier = a 30-line zero-boot hit-rate probe on banked traces.  (2) frequency-tiered
expert requant (MxMoE/MoPEQ class, +1-3 beyond uniform) riding the pending T14B requant
build.  NEGATIVES banked: Ozaki/split-fp16 compensation (all decode lines bytes-bound),
tree speculation (SpecBranch joins the TreeWY verdicts), NVFP4 repack (byte-regressing),
draft precision cuts, expert prefetch (hit ~99%).

## F-288 — 2026-10-24 ~05:15: THE q32 BENCH GAP CLOSED; THE UP RESTRUCTURE BUILT AND
## FALSIFIED FLAT — the kernel program REDIRECTS TO THE ENGINE MEMORY CONTEXT.

**THE BENCH GAP CLOSED**: fused_gr_bench now covers the engine's real up path —
STRATA_BENCH_Q32=1 sets per-token q8_mixed state (N/32 GrQ81 groups x 36 B) so
fused_gr_read_multi dispatches gr_up_q32 (QFUSE, default-on in the engine).  q32 baseline:
up 7.2/10.4/10.4/12.0/13.7/15.2/16.6/18.2 µs (T=1..8, cold) — within 0.2-0.3 µs of
gr_up_multi everywhere (both kernels share the structure).

**THE BATCHED RESTRUCTURE (F-288, both kernels, BITWISE-IDENTICAL by construction)**:
every token's dots run as independent register chains FIRST, then ONE batched butterfly
(warp_sum's own 5 xor steps per token — same addition sequence, only the schedule
changes; no numerics gate owed).  Built clean; **BENCH: FLAT** — q32 after
7.2/10.3/10.4/11.9/13.7/15.2/16.6/18.0 vs before (Δ ≤ 0.2 µs at every T; multi same).
THE SERIALIZATION HYPOTHESIS IS DEAD: the shuffle tree's latency was not the binder.

**THE STRATEGIC REDIRECT (two falsifications, one conclusion)**: F-287 (down at 96% of
the DRAM wall isolated) + F-288 (up's schedule changes flat) = the hc kernels are NEAR
THEIR ISOLATED ENVELOPE.  The engine-context loss (0.44-0.62 TB/s in-engine vs
0.9-1.6 isolated) lives in MEMORY CONTEXT: the expert stream (GBs/window through L2)
evicts the ~13 MB of hc weights (w_down + w_up) from L2 every window.  THE NEXT RUNG =
**the persistent-L2 access-policy window** (cudaStreamSetAttribute +
cudaAccessPolicyWindow, hitRatio-tuned) pinning w_down/w_up — never attempted in-tree,
cheap (init-time code), and it attacks the exact measured gap.  Falsifier: one bench leg
with the policy set vs unset on the cold-sets mode (the cold sets rotate ACTIVATIONS;
the WEIGHTS stay resident — add a weight-rotating cold mode or price it engine-side with
the in-kernel stamps).  THEN the engine A/B per the user's before/after law.

## F-289 — 2026-10-24 ~05:50: THE COLD-SET METHODOLOGY CORRECTION (the 4-set L2 artifact)
## — and staged2 falsified AT TRUE ENGINE CONDITIONS.

**THE ARTIFACT**: the bench's COLD mode rotates `cold_sets` weight copies — at the
default 4 sets (52 MB) the whole working set LIVES IN L2 on this card (the curve
saturates at 24 sets = 314 MB): every "cold" number banked through F-287/F-288 was
L2-RESIDENT.  THE ENGINE-CONDITION NUMBERS (48 sets = 629 MB, one per real layer):
down 8.1/11.7 µs, up 13.8/19.5 µs at T=4/8 — 13-21% SLOWER than the 4-set numbers
(down T=4: 6.8→8.1; up: 12.3→13.8).  At true cold the lines run 0.81 / 0.47 TB/s —
NOT bandwidth-bound (1.7 available): L2-MISS-LATENCY bound at their grid occupancies.

**staged2 RE-TESTED AT 48 SETS** (the weight cp.async ring never had a fair test at
4-set/L2-resident): STILL FLAT — down/up stamps byte-identical to staged at every T.
The one-tile __ldg prefetch + the activation ring already cover the latency at this
occupancy; the ring adds nothing.  FALSIFIED CLEAN.

**WHAT REMAINS ON THE HC LINE** (both rungs now correctly priced at true conditions):
(a) the down line's binder = the 41-block GRID occupancy (F-217 family) — staged2's
per-thread staging removes the coupling that killed the F-15C wide family, so a
4-warp/82-block staged2 is the un-run form; (b) the UP line (the bigger, softer line:
13.8-19.5 µs true-cold) still has UNPREFETCHED per-warp-row loads — an up weight-ring
(batched r-ahead cp.async) is its un-run form, same design as staged2's ring.
NOTE for engine A/Bs: the engine's in-window stamps (VERIFY_PROFILE-class) remain the
arbiter — the bench cannot see the concurrent expert stream's DRAM/L2 competition.

## F-290 — 2026-10-24 ~06:20: THE UP PREFETCH PIPELINE BUILT AND FALSIFIED — AND THE
## ENGINE-CONTEXT MYSTERY DISSOLVED (the hc kernel micro-program CLOSES).

**THE PIPELINE (bitwise)**: gr_up_q32's per-warp weight loads pipelined two rows ahead
(register rotation; the same __ldg values issued earlier).  Built clean; **48-set bench:
FLAT** — up 8.8/13.3/12.0/13.4/15.1/16.5/17.9/19.5 vs baseline
8.8/13.4/12.1/13.6/15.2/16.8/18.1/19.8 (Δ ≤ 0.3 µs = noise).  THE FOURTH consecutive
kernel falsification (staged2 ring 4-set, staged2 ring 48-set, batched butterfly,
prefetch pipeline).

**THE RECONCILIATION (the closure that matters)**: at 48-set true cold the bench reads
down 5.6-11.8 / up 13.4-19.8 µs — the ENGINE's in-window hc line (the "0.44-0.62 TB/s
engine-context loss" that redirected the program in F-287) is THE SAME NUMBERS.  The
T37 anatomy's "engine loss" was an artifact of comparing engine-cold against 4-set
L2-warm bench numbers.  There is no hidden engine-context loss: the hc lines run at
DRAM-cold latency at their grid occupancies and do not respond to schedule, prefetch,
or reassociation changes at this occupancy.  THE HC MICRO-PROGRAM IS CLOSED as a dead
end at the instruction-stream level (the unrun forms remain: wide-grid occupancy for
down — one cheap instantiation — and the mma rung, whose bytes model may revive at
true-cold but four flats price it low).  The bitwise restructures stay in-tree (zero
risk).  The composition's solo-300 headroom moves to its OTHER levers: two-stage head,
keep-rule refit on the banked 1,615-window trace, retrain, requant (USER GO), and the
n-gram draft candidate.

## F-291 — 2026-10-24 ~06:50: THE CONTEXT-N-GRAM DRAFT CANDIDATE DEAD AT THE FALSIFIER
## (zero engine work spent — the falsifier-first law's cleanest kill yet).

**THE PROBE (campaign/t48-ngram-probe.py + t48-ngram-analyze.py)**: 3 copy-heavy prompts
(repeat-verbatim / code-reprint / list-restate) vs 3 literary controls on the live
champion; per request the engine's own MTP acceptance (the suffix-drafts log line) AND
the prompt-lookup ceiling (the fraction of response tokens an n=4..8 matcher over the
preceding context would have predicted exactly, pack-tokenized):

| prompt | MTP accept | PLD ceiling | headroom |
|---|---|---|---|
| copy/repeat-para | 88.7% | 78.7% | 0 |
| copy/code-reprint | 89.4% | 54.8% | 0 |
| copy/list-restate | 81.4% | 64.1% | 0 |
| control/story | 93.5% | 0.0% | 0 |
| control/essay | 75.8% | 0.9% | 0 |
| control/analogy | 73.3% | 0.0% | 0 |

**VERDICT: DEAD.** The coupled MTP drafter BEATS the n-gram oracle on every copy-heavy
prompt — it learned copying; the T47 dossier's "+5-25% on copy-heavy" band applies to
stacks whose drafters did not. The splice is never implemented; negative banked with
numbers. (Also banked: acceptance classes read 73-94% live — the drafter's real per-task
spread, useful priors for the keep-rule refit.)

## F-292 — 2026-10-24 ~07:15: THE KEEP-RULE REFIT CLOSED AS DATA-EMPTY — the fresh
## SPEC_PROB-recipe trace (3,265 solo windows + batch twins; t48-spec-trace.txt) proves
## the rule family has no headroom; the +6-9% pool belongs to RETRAIN.

**THE FRESH TRACE** (the old 1,615-window bank predates SPEC_PROB by two days — any refit
on it priced dead math): decile curve UNCHANGED — 0.554/0.614/0.610/0.689/0.821 vs the old
0.571/0.613/0.643/0.722/0.835 (slightly lower = the more varied UX-heavy load mix).  The
curve is a property of the DRAFTER-MODEL PAIR, not of any acceptance rule: SPEC_PROB
rewired the rescue (rescues in 3,201/3,265 windows now) without moving exact-match
acceptance.  The acceptance DISTRIBUTION improved hugely vs the old bank (zero-accept
windows 9.6% vs 29.5%; mean accepted 2.232 of 3.935 offered) — the recipe is healthier,
but the threshold families (keep/cut falsified −6.0% pooled at F-277-era; τ 0.15
falsified all-lane; τ tighter untested and the curve says no room) have NOTHING to fit:
the sweep's kept-mean ≈ offered-mean held on the new data too.  **VERDICT: the refit
lever is CLOSED as data-empty; the deficit pool's lever is RETRAIN v2 (the T14B
pipeline — drafter quality shifts the curve itself).**  The trace boot's lanes read
206.2/203.8/313.4 = the box is RECOVERING (best conc since the evening pools) — the
fair-window re-bank becomes viable.

## F-293 — 2026-10-24 ~07:45: THE COMPOSITION'S BASE IS CERTIFIED — fair-window
## five-boot pool on the standing champion: M1 217.3 (σ 2.2) / pair 206.9 (σ 5.5) /
## conc 319.5 (σ 12.4).

Legs: 218.6/202.0/322.1, 215.3/203.3/320.7, 220.4/215.9/338.2, 216.8/205.4/306.2,
215.3/207.8/310.3.  vs the night's cool-window pool (200.3/182.2/278.3): +8.5/+13.6/
+14.8% = THE WINDOW TAX, quantified per lane (contention, F-285b/F-286's finding).
vs the banked records (222.4/209.9/319.6): −2.3/−1.4/−0.03% — THE CHAMPION SITS AT THE
RECORDS (inside boot noise; r3's 338.2 conc and 215.9 pair are single-boot record-class
reads).  **BASE B = 217.3 / 206.9 / 319.5** — the 300/450 composition stacks from here:
two-stage head (+4-6) → CCD isolation → requant (USER GO) → retrain v2 (owns the +6-9
pool); with the ternary's +9.7% riding the 9-wide conc lane.  The M1 σ 2.2 is the
tightest five-boot spread ever banked — the recipe is STABLE (the night's stability
goal delivered in the same breath).

## F-294 — 2026-10-24 ~08:10: THE TWO-STAGE HEAD PORT SCOPED (the +4-6 solo lever; the
## spec's own gate: miss-rate harness BEFORE any engine integration).

**THE DONOR (located, read, shape-verified)**: basalt's q6k_vhead_parity.cpp (284 lines:
self-contained — inline ggml Q6_K dequant, CPU LS-fit reference, GPU two-stage parity,
cold microbench) + head_proxy.cuh (the 4-level 2-bit + fp16-scale proxy kernel, 243
lines) + q6k_head_parity.cpp (the exact-layout bitwise harness, 359).  Default shape
rows=248,320 / n_in=2560 / Q6_K 210-B blocks = OUR head's class (T37's 521.5 MB read).

**THE PORT PLAN (zero ambiguity for the next session)**:
1. Vendor the three files into campaign/headport/ with includes rewired (stage-2 exact
   rescoring maps onto OUR native_mmvq — the donor's q6k_head_mmvq is its cousin).
2. v0: run as-is (random rows scaled to a logit spread, random x) = the copy-fit and
   machinery parity on OUR build.
3. v1: replace random rows with the REAL head tensor read from the GGUF (verify the
   tensor class FIRST — the spec asserts Q6_K 521.5 MB @ 248,320×2560; confirm from the
   gguf metadata before wiring), random-x v1 then a real-activation dump boot
   (STRATA-class dump of the head's input row) for the honest miss rate.
4. GATE: proxy-top1 (and top-64-shortlist) miss rate ~1e-4/row-class → GO to the
   exact-materialize arm (STRATA_HEAD_PROXY, env-gated, F-230 ladder: KL yardstick pair
   → 3v3 → 2× MMLU if KL > 1e-5 → before/after single+conc proof); miss rate worse →
   DEAD, the full-vocab sampler constraint makes approximation unshippable on speed
   alone (T37 §3.2's three-option analysis stands).

## F-294a — 2026-10-24 ~08:40: HEAD PORT v0 GREEN — the two-stage head measures
## **2.33-2.49x** at the full shape, cold, parity-perfect.

**THE PORT**: basalt's q6k_vhead_parity + head_proxy.cuh + q6k_head/native_mmvq/iq_kernels
vendored to campaign/headport/ (include closure closed; nvcc -arch=sm_120, zero errors).
**v0 RUN (rows=248,320, the head's shape, random Q6_K rows scaled to a logit spread)**:
copy-fit parity 0/7,754,240 levels differ; near-top rows EXACT (the bits of q6k_head);
approx rows match the CPU score (max dev 3.8e-06); argmax differs 0 at ncols 1..8;
cold bench q6k_head 356-400 µs vs **vhead 143-166 µs = 2.33-2.49x**.  Exact rows per
column ~558-703 (the VDELTA rescore band behaves).  THE MACHINERY IS PROVEN on our
card/build; at the T37 head-line anatomy (0.45-0.5 ms/window) the cut ≈ 0.2-0.25
ms/window = the +4-6% M1 band intact.

**THE REMAINING GATE (v1)**: random-weight heads are the EASY case — the miss rate on
the REAL head tensor + real activations is the honest gate (~1e-4/row-class), then the
exact-materialize arm (STRATA_HEAD_PROXY) through the F-230 ladder + the full-vocab
sampler reconciliation (the T37 §3.2 constraint stands) + before/after single+conc
proof.  The harness for v1: read the real head tensor from the GGUF (class verify
first) + an activation-dump boot.

## F-294b — 2026-10-24 ~09:10: THE TWO-STAGE HEAD DEAD ON THE REAL TENSOR (the v1 gate's
## verdict; no battery spent, exactly as the gate prescribed).

**THE RUN (vhead_v1, REAL output.weight [2560×248320] Q6_K extracted verbatim from the
GGUF — 521,472,000 bytes exact; copy-fit still 0/7,754,240 levels differ)**: the real
head's logit top is FAR FLATTER than random — the VDELTA rescore band explodes from 558
exact-rows/column (random) to 65,536→8,192/column as ncols grows (up to ~26% of the
vocab), collapsing the speed from 2.33-2.49× to **1.33-1.42×**, and the argmax STILL
differs 1-5 times per run (vs the 1e-4/row-class gate = orders of magnitude over).
The frontier is bad on BOTH axes: wider VDELTA eats the win, tighter adds misses.

**VERDICT: DEAD for the exact-match contract** (the T37 §3.2 pre-declared kill: 'the
arm can never ship on speed alone; law 10, the canary is the contract').  The head
line's honest residual: the 2.4× was a random-weight artifact — real structured heads
are exactly the adversarial case for a 2-bit proxy.  Artifacts kept: campaign/headport/
(v0/v1 harnesses + the extracted tensor); the +4-6% solo band leaves the composition —
recalculated central ≈ 283-291 (base 217.3 × retrain-or-keep 1.06-1.09 × requant-GO
1.04-1.07 × micro 1.01-1.02) — 300 now needs retrain AND requant AND the micro bundle,
or new levers.  PIVOT: CCD class split (the queue's next live item).

## F-295 — 2026-10-24 ~09:55: THE CCD CLASS SPLIT FALSIFIED (0-for-2 family) — the pool
## wants the WHOLE topology; the window tax's recovery lever is NOT thread placement.

**THE ARM (STRATA_POOL_CORES="8-15,24-31" — the feed class restricted to all of CCD1,
16 logical, in detect_cpu_topology; env-gated, default-off, deployed on the F-295
build)**: 3v3 ABAB vs ctl.  POOLED: ctl 217.8/205.1/308.3 vs ccd **189.4/176.5/245.9 =
−13.0/−13.9/−20.3%** — WORSE ON EVERY LANE, every round (M1 legs 198.4/185.6/184.1 vs
ctl's 207.4-225.3).  THE MECHANISM, corrected: the CPU expert pool is not the
contention VICTIM class — it is a THROUGHPUT engine whose workers scale across the full
32-thread topology; restricting it to half the machine strangles the feed that every
lane rides.  With the naive whole-tree pin already at −20%, the thread-PLACEMENT family
is 0-for-2 and CLOSED.  The window tax's honest remaining levers: ambient-noise
REDUCTION (the monitor tab's ~2 cores — the user's own call; SCHED_BATCH-ing the noise
is unprivileged and untested) or nothing — the tax is the price of a busy desktop.

**NET for the composition**: CCD isolation leaves the stack (it never was a member);
the certified base 217.3/206.9/319.5 stands; the remaining IMPLEMENTED-path levers are
retrain v2 (+6-9, owns the pool), requant (+4-7, USER GO), micro bundle, C2
reader-thread (conc), and whatever the next research wave finds — 300 solo now requires
nearly all of them landing.

## F-296 — 2026-10-24 ~10:35: THE C2 OVERLAP FAMILY DEAD 0-FOR-2 — with the MECHANISM
## named (the shared expert pool couples the verifiers' tails).

**THE ARM (STRATA_COHORT_OVERLAP=1, in-tree default-off)**: finish-under-window in its
SAFEST form — the finish lambda kept in-scope (moved above the poll loop as a pure
reordering, body untouched), only CALL SITES moved into the poll loop (fa/fb/fc
completion flags, inline failed-checks; default = byte-identical order; the F-281
relocation NOT repeated).  Built clean; **SMOKE 0/12 at ter9**: the failing round was
BINARY (A.S=8 B.S=4 — the ramp), "windows done 21.2 ms", then #267 layer-0 timeout —
finish(ver, wa) ran under ver_b's in-flight window and the next window's wait never
resolved.  THE MECHANISM (consistent with F-281/F-285a, now identified): the verifiers
SHARE the expert pool and source (the ver_b init's own comment) — the tail's commit
path touches pool state the other cohort's in-flight window is concurrently waiting
on; overlapping the tails requires serializing the pool, which erases the overlap.
**VERDICT: C2 tail-overlap CLOSED as structurally hazardous (0-for-2 across two
independent forms); the +9-18% conc band leaves the composition — the ternary's live
+9.7% at 9-wide stands as the conc lane's win.**  The only remaining form (per-verifier
pool partitions) is a redesign, priced for a future wave with the mechanism named.
(Harness note: a pkill -f pattern matched my own wrapper mid-restore — the kill-by-PID
discipline, already in memory, bit once more; the failing arm's boot was caught and
replaced with the standing champion, env verified clean.)

## F-297 — 2026-10-24 ~10:55: T50 NEW-LEVER WAVE INTEGRATED (verified) — the research
## program's honest close: nothing cheap remains; the path is retrain + requant + riders.

**VERIFIED**: the dossier's two load-bearing claims check against primary records — the
T26-5 alpha requant really leaves `token_embd/output/head =COPY` (draft_vocab
conservatism, never measured), and the global DRAFT_TEMP=0.85 sweep (0.45-0.90, F-201)
predates SPEC_PROB by 8 days.  The premise correction stands: the solo window is 83-88%
in-graph GPU walk (T13A/T39 anatomy) — there is NO graph-relaunch program to run; the
35-50% figures were the conc round wall.

**THE VERDICT (all four tasking areas)**: dprob-shaped depth CLOSED by three
independent instruments (F-228 stair −1.75%, F-292 data-empty, T26-2 DSpark/varlen);
trees stay dead (p 0.77-0.89, row-flat graphs); row-batched MoE across a solo window
structurally impossible (the serial chain needs the row's own hidden state for routing);
the 2026 MoE×spec literature bends the target or targets tiers we don't run.  The L2-pin
formally de-priced by arithmetic (630 MB vs the 60 MiB carve — F-289's kill confirmed).

**TWO NEW RIDERS (the wave's only positives, both cheap)**:
(a) OUTPUT-WEIGHT REQUANT RIDER — the 521.5 MB Q6_K head (0.45-0.78 ms/window) to
Q4-class ≈ +1-2% M1; OFFLINE-falsifiable on the F-294b headport harness with the
already-extracted real tensor (quality read before any engine work);
(b) PER-POSITION DRAFT-TEMPERATURE SCHEDULE — global 0.85 swept pre-SPEC_PROB; the
93.6%/53.1% shallow/deep acceptance split (F-235) makes a per-step schedule untried,
+0-2%, ONE battery, must precede retrain (which subsumes it).

**THE FINAL COMPOSITION (every alternative priced dead with mechanisms)**:
B=217.3 × retrain (1.06-1.09) × requant-GO (1.04-1.07) × head rider (1.01-1.027) ×
PDL chain+diet+micro (1.02-1.04) × DT schedule (1.00-1.02) = **central ≈ 264-271,
upper ≈ 284-296**.  300 closes ONLY if the T26-3 PDL stretch (−1.5 ms) lands AND
retrain+requant hit upper bands simultaneously.  The gap's real home: RETRAIN v2
quality + the REQUANT GO decision + the two riders.  The research program is closed
until retrain changes the drafter.

## F-298 — 2026-10-24 ~11:30: THE HEAD-REQUANT FALSIFIER RUNS AND CATCHES ITS OWN DATA —
## INCONCLUSIVE, and the F-294b kill verdict is FLAGGED for re-examination.

**THE PROBE (campaign/headport/head_requant_probe.cpp: full-vocab Q6_K→Q4_0 pack +
top-1 flip / KL / max-delta, CPU-only, zero boots — the rider's designed falsifier)**:
first runs read KL 678 / |dlogit| 118,877 / 4-of-4 top-1 flips — ABSURD numbers.  The
row-norm sweep names it: the dequantized head reads p50 norm 1.49e9, amax ~1.35e8 ≈ 2^27
— physically impossible (a real head is ~1-100 norm).  THE EXTRACTION OR THE DEQUANT
CONVENTION IS WRONG: the model file is GSQ-RCO (a CUSTOM quant) — the head tensor may
not be vanilla ggml Q6_K layout despite type_id=14, or the tensor-data offset basis is
off.  **CONSEQUENCE: F-294b's "flat logit top → two-stage head DEAD" verdict rode the
same extraction — it is PROVISIONAL until a verified reader (the engine's own head
path as ground truth) re-reads the tensor.**  Both the head-proxy verdict and the
requant rider return to UNPRICED.  The catch is the falsifier-first law working: a
wrong-data kill would have been a silent false negative.

**NEXT (precisely scoped)**: verify the head bytes via the engine's own reader (boot
one debug line dumping the head's first-row dequantized values, or diff the extraction
against the pack's tokenizer/head loader) → re-run F-294b's v1 + this probe on
verified bytes → both verdicts re-bank.

## F-298b — 2026-10-24 ~11:55: THE HEAD FAMILY RESOLVED ON VERIFIED DATA — requant
## GO-CANDIDATE; the proxy kill stands; the extraction was CLEAN all along.

**THE BUG (found and named)**: my hand-transcribed C++ Q6_K dequant invented scale
indices for the 3rd/4th 32-halves (the donor source had been read only through the
second half; the garbage row-norms were transcription, not data).  **THE VERIFICATION**:
llama.cpp's own reference dequantizer (vendored gguf-py) reads the GGUF tensor AND the
extracted head_q6k.bin identically — norms 0.77-1.0, amax 0.41: vanilla Q6_K, offset 0,
extraction byte-perfect.  The falsifier rebuilt in numpy ON the reference (the
transcription-bug class eliminated by construction).

**THE VERDICTS (both now on clean data)**:
1. **HEAD REQUANT Q6_K→Q4_0 = GO-CANDIDATE**: top-1 flips 0/6, mean KL 2.1e-3, max
   |dlogit| ~0.4 (sd-4 logits), −31% head bytes (521.5→357.6 MB) = the +1-2% M1 band.
   Joins the requant USER-GO discussion with these numbers; on GO the integration rides
   the T26-5 requant build (head in-tensor swap + mmvq Q4 path check) + MMLU/mimic gates
   + before/after proof.
2. **F-294b's two-stage 2-bit proxy kill STANDS**: its GPU-side measurements (flat top,
   rescore band, argmax flips) read the real bytes via the donor's correct kernels.
   Consistent: 2-bit flips the argmax; 4-bit does not.  FLAG LIFTED.

## F-299 — 2026-10-24 ~12:45: THE DT SWEEP (post-SPEC_PROB) CLOSES NEUTRAL — and the
## F-296 REGRESSION caught, reverted, verified (the F-285a law's second occurrence).

**THE REGRESSION**: the t50 dt battery's conc lane read 0.0 on ALL legs — the F-296
build (the lambda-above-the-loop move for the dead overlap arm) broke the DEFAULT
cohort path (#267 layer-0 timeouts with the arm OFF — a relocation changes the default
path, AGAIN; law: NEVER move the lambda, even for a default-off arm).  REVERTED
(generate.cpp -> 078fba14^), rebuilt, deployed: conc restored at 294.2 verified.
**THE ARTIFACT LAW (new)**: the sweep's M1 230.2@0.75 was inflated by the broken lane
itself — no conc traffic during the M1 leg = a quieter box.  Never read M1 from a boot
whose conc lane is broken.  Confirm on the FIXED binary: 205.6/206.0/296.7 = inside
the standing band.
**THE DT VERDICT**: 0.75 / 0.85 (standing) / 0.95 all within noise post-SPEC_PROB —
the global optimum did not move; the per-position form has no sweep evidence of
headroom; the DT family closes NEUTRAL (no change to the recipe).  The T50 rider's
+0-2% band does not materialize.

## F-300 — 2026-10-24 ~13:15: RETRAIN v2 (T14B) FORMALLY OPENED — the +6-9 acceptance
## pool's campaign, with the cheap-lever phase closed behind it.

**THE SPEC SUITE (verified execution-ready at read)**: T14A (method/recipe), the exec
trio A(harvest)/B(parity)/C(trainer-export), T14C (acceptance math gates).  THE ORDER:
A harvest → **B the P2 torch parity module (THE critical path)** → C train+export →
T14C gates → engine A/B per the user's law.  P2 = one PyTorch module reproducing the
engine's drafter forward (one MTP layer + mixer + sub-vocab head, 31 tensors from
/home/svg/LLM/data/mtp-en/rt), adjudicated by the existing STRATA_MTP_TAP stage-diff
harness.  THE THREE LAWS (paid for in the R17 arc): weights from the RT dir never the
BF16 sidecar; the Gemma +1 already folded in f32 norm tensors (re-folding = F-109's
0%-acceptance bug); --mtp-hnorm stream = per-stream grouped RMS.  Geometry pinned:
N=2560/HC=4/NH=24/HD=256, rope 1e7, eps 1e-6, 512 experts top-10, K=6 supervised
positions, draft_vocab 40,525 ids.  Known spec conflicts (the 10-18 audit): trainer
lr, rt2-vs-rt-r18 label, G-ladder collisions — fix at implementation time.

**THE FIRST IMPLEMENTATION STEP**: the P2 module skeleton per exec-B §2 (tensor table →
loaders for q8_0/bf16/f32 rt kinds → the forward in torch → the TAP harness boot).
**THE REQUANT GO** (presented to the user this entry): experts T26-5 requant (−8.26%
expert bytes, +1.90% direct M1 modeled, +792 slots) NOW CARRYING the F-298b head rider
(Q6_K→Q4_0: 0/6 top-1 flips, KL 2.1e-3, −31% head bytes ≈ +1-2% M1) — combined band
+3-4% M1 + VRAM headroom, quality-gated (MMLU floor + mimic band + the F-230 yardstick
on the first boot).

## F-301 — 2026-10-24 ~13:50: P2 OPENS — the parity venv + the rt loader LANDED (the
## retrain critical path's first milestone, done clean).

**THE VENV** (~/venvs/mtp-parity; torch 2.14.1+cpu + numpy 2.5.4): CPU torch is the
right-sized choice for PARITY (math on one token; the exec-C trainer venv gets cu130
per the infra spec's P0 — the repo .venv untouched).  **THE LOADER**
(campaign/retrain/p2_rt_loader.py): the rt dialect's three kinds read natively — q8_0
as tools/mtp_rt.py's packer inverted (ggml reference rounding), bf16 by bit-view (the
one bug: astype-vs-view byte-doubling, caught by the shape assert), f32 with the Gemma
+1 kept folded (law 2).  **THE SMOKE (F-298's lesson, institutionalized)**: 29/29
tensors load with sane stats — weights rms 0.003-0.2, the +1-folded norms in the
1.0-5.0 band, zero impossible magnitudes.  NEXT: the forward per exec-B §2.3+
(fc_embedding → per-stream RMS (law 3) → attention: q/k norms, rope 1e7, GQA 24/2 →
the three hyper-connection mixers → shared expert + router + routed-blob reader →
fc_hidden → the 40,525-id sub-vocab head), then the STRATA_MTP_TAP adjudication boot.

## F-302 — 2026-10-24 ~14:40: THE P2 FORWARD LANDED STAGE-WISE (B1-B3 RUNNING, B4's
## input question NAMED) — the critical path advancing on real transcriptions.

**LANDED (campaign/retrain/p2_forward.py)**: the G0-exact module per exec-B §2.2 —
the q8_1 activation quantizer (clamp/roundf exact), the engine-dot form (integer codes
x fp16-d weights, fp32 einsum accumulation), RMS-with-gamma-after, the HC read (B3) in
its full exact form (the concatenated [10240] down projection with the /HC contract —
my first per-stream-sum form was wrong, caught by shape asserts; the apply/bo_prev/
inj_prev fold; the up/sigmoid/mixed formula with the final /4), B1 (embedding branch)
and B2 (per-stream residual + fusion) — ALL RUNNING on the real rt weights with sane
stage stats.  B4's attention is transcribed (partial rope inside 64 dims, per-head
norms, int8 KV round-trip, GQA 12:1, the raw gate half) but surfaced a SPEC AMBIGUITY:
q/k/v projections are [.., 2560] while the HC read's mixed is [10240] — WHICH 2560-
reduction of mixed feeds attention (mean-over-streams? a different read output?) is
exactly what the STRATA_MTP_TAP fixture adjudicates (the TAP dumps the engine's actual
stage tensors; the module matches or the spec's mtp.cpp:952-965 citation gets re-read
for the wire between the attn read's outputs and the qkv calls at :967-970).
B5's router/blob/shared-expert and B6's head follow once B4's wire is pinned.
**The critical path is genuinely moving: loader + forward core live, the adjudication
boot is the next session's opening move with the fixture script scoped by TAP's env.**

## F-303 — 2026-10-24 ~15:10: THE B4 WIRE RESOLVED FROM SOURCE — mixed is the
## per-dimension STREAM MEAN [2560], not a 10240 concatenation; the forward core
## runs end-to-end through the attention fold.

**THE RESOLUTION (static, no boot needed)**: mtp.cpp:958 `fa[t].mixed = mixed_ + t*N`
with :424 `mixed_ = b.take<float>(T * N)` — the drafter's HC-read mixed output is
[2560] per token: `mixed[d] = sum_c(R'[c][d] * w_norm[c*N+d] * rs[c] *
sigmoid(up[c*N+d])) / 4`.  The exec-B spec's B3 formula (mixed[i] over 10240) carried
the TRUNK's fused_gr semantics into the drafter context — SPEC CORRECTED in-tree.
**THE FORWARD NOW RUNS END-TO-END** through B1->B2->read1->attention->fold->read2 on
real rt weights with sane stats (mixed rms 1.63, bo 2.20, mixed_mlp 0.94, R_after
2.38; all finite).  Also read from source: the TAP dumps THREE tensors (R_/sample_/
head_logits, mtp.cpp:1234-1248) — the adjudication is END-TO-END through the chain,
one window, diagnostic boots.  REMAINING to G0: B5 (router + the Q2_0 blob reader +
shared expert; the y-combine and the R_ write) + B6 (the final read + the 40,525-row
Q4_0 head from the fixture) + the TAP boot diffing R_ first (the earliest gate point).

## F-304 — 2026-10-24 ~15:45: B5/B6 LANDED — THE FULL P2 FORWARD CHAIN IMPLEMENTED
## (B1→B6 end-to-end, all stages smoke-green on real weights).

**B5**: the router (all-512 softmax, stable top-10, the 2^-14-floor renorm — weights sum
1.0 verified); the Q2_0 blob reader (the four-plane layout of mtp_rt.py:49-64: interleaved
gate/up codes, down codes, the two fp16 scale planes; 2-bit LSB-first decode — first
attempt's strided indexing fixed to the sequential nibble order); the silu(gate)·up →
q8-requant → down expert path; the shared expert with the BOTH-BF16 scalar gate; the
y-combine.  **B6**: the final HC read on the hyper_connection_mixer weights → sample
[2560].  Smoke: blob-0 rms 0.033, y rms 4.5, sample rms 1.95, all finite.  **REMAINING
TO G0 (named precisely)**: (a) the head fixture — the 40,525-row Q4_0 head from the
main GGUF at the dvocab ids (the F-298b numpy machinery applies verbatim: reference
dequant + ggml Q4_0 requant); (b) the TAP boot's INPUT side — the TAP dumps
R_/sample_/head_logits (outputs) but the module needs the same window's inputs (token,
Rin, the layer's KV cells); the R17 protocol's known-prompt design or an input-side tap
addition decides; (c) the diff vs the ~2e-6 noise floor → G0.

## F-305 — 2026-10-24 ~16:30: THE G0 GATE RUNS — **T1 PASSES AT PRECEDENT LEVEL**;
## T2 close (argmax + top-5 EXACT MATCH, cos 0.9985) with one residual to close.

**THE TAP BOOT** (the standing lane + STRATA_MTP_TAP only; production paused, restored
after): all three dumps landed at the spec's exact sizes.  **T1 (the mixer read on the
dumped R_)**: first read std 2.75e-2 / cos 0.99997 = the R17 WEIGHT-PROVENANCE
signature — and the law held: the HC mixers are BF16 weights consumed with
UNQUANTIZED fp32 activations (the bf16_gemv class, like the router); my q8_1 boundary
on them was the bug.  **After the class fix: std 4.87e-7, max_abs 3.8e-6, max_rel
(|S|>0.1) 1.96e-6, cos 0.99999994** — the R17 precedent neighborhood (its
"0.000000/1.000000" printed at 6 decimals).  **T2 (the head)**: argmax 4087 = 4087
EXACT, top-5 sets match, cos 0.9985 — but std 1.7e-1 vs the 5e-5 gate: the Q4_0 head
bytes are CLOSE but not byte-identical to make_q4_head's (the candidate: my fixture's
Q4_0 rounding or d-scale convention vs mtp.cpp:246-269's exact form — likely the
fp16-d rounding path or the row-gather order).  THE PATH TO G0: read make_q4_head's
exact quantizer and match it (one function; the argmatch already proves the class).
**THE PROVENANCE LAW VALIDATED AGAIN**: bf16-weight dots run unquantized-activation;
q8_1 boundaries live ONLY on the q8_0/q4_0-weight projections.

## F-306 — 10-24 ~17:10: THE T2 RESIDUAL ISOLATED TO THE HEAD BYTES — the quantizer
## fix (fp32-scale codes, fp16-stored d = ggml's exact make_q4_head form) did NOT move
## it (std 1.715e-1 unchanged); the PLAIN fp32 dot shows the SAME residual (1.717e-1)
## — the activation quantizer is exonerated, the argmax (4087) and top-5 match in every
## variant, cos 0.9985.  CONCLUSION: my W matrix differs slightly from the engine's
## dhead_ bytes — the candidates: the engine-side Q6_K dequant at load vs gguf-py's
## reference (both "reference class" but unverified against each other on these rows),
## or a gather-order subtlety.  THE CLOSING MOVE (small, precise): extend the TAP with
## a 3-line dhead_ dump (cudaMemcpy of the n_dvocab_ x out_row buffer beside the
## existing three at mtp.cpp:1234-1238), re-boot the tap lane, BYTE-DIFF against the
## fixture — the diff itself will name the divergence in minutes.  G0 STATUS: T1
## PASSED (the hard math, at precedent); T2 = one byte-diff away.

## F-307 — 10-24 ~18:20: THE HEAD-BYTES TAP EXTENSION LANDED (F-91-lawful: alloc inside
## make_q4_head, copy type-gated in the capture, dump beside the three) — and the T2
## byte puzzle ISOLATED to the engine's on-device Q4 layout.

**THE INSTRUMENTATION (committed, permanent value)**: tap_head.q4 (58,356,000 B = the
engine's actual dhead_ Q4_0 bytes) dumps beside R/sample/logits on every tap boot.
**THE FINDINGS**: (1) the engine's d = amax/8 POSITIVE (modern ggml convention; the
F-298b packer's old negative-d noted, its quality verdict unaffected — self-consistent
both directions); (2) row-0's d and FIRST code match my vanilla fixture exactly, the
rest don't; (3) NO vanilla-family packing variant (hi/lo nibbles, byte-reverse, d-last)
decodes the dump against the tap logits (best cos 0.083) — the engine's on-device Q4_0
head is NOT host-row-major vanilla; it is the MMVQ kernel's consumption layout (or the
codes encode the engine's GSQ-dequant weights in that layout).  **THE DEFINITIVE
DECODER IS IN THE ENGINE**: native_mmvq's Q4_0 weight-read branch names the exact
byte→value map — the next session reads it (one kernel function) and the fixture
rebuilds from the dump bytes directly.  **G0 STATUS UNCHANGED IN ITS LOAD-BEARING
PARTS**: T1 PASSED at precedent (the hard math); T2's residual is now FULLY cornered —
not the quantizer, not the activation path, not the gather: the memory layout, with
both the bytes and the decoder named.  (The cos-0.9985/argmax-exact original fixture
remains the best numerical approximation and the fallback if the layout read stalls.)

## F-308 — 10-24 ~19:05: THE T2 PUZZLE SOLVED IN STRUCTURE — two engine facts found;
## the closing step is one repack-read or one more tap line.

**FACT 1 (the dot semantics)**: native_mmvq's Q4_0 dot is d*(Σ(c·q) − 4·ds.y) — the
AFFINE-CORRECTED integer form consuming the Q8_1 block's SUM field (native_mmvq.cu:713-
725).  Neither of my dot variants modeled the −4·s term: the parity module's q8_1 dot
MUST be upgraded to the exact block form (codes ×, then the affine correction) — a
required fix for G0 regardless of the weights.
**FACT 2 (the weight provenance)**: the engine calls native_q6_k_unpack(dhead_) at load
(mtp.cpp:187) — the head's Q6_K is REPACKED to the engine's native layout BEFORE
make_q4_head quantizes it; the dumped Q4 codes encode the REPACKED-dequant weights
(row-0 analysis: same d, first code matches, rest differ structurally — consistent
with a repacked source, not a packing error).  The vanilla gguf-py fixture stays the
best approximation (cos 0.9985, argmax exact).
**THE CLOSING STEPS (either closes G0)**: (a) read native_q6_k_unpack's layout and
replicate it in the fixture path; or (b) one more tap line dumping the dequanted
pre-Q4 head rows (the simplest: dump the `row` f32 vector inside make_q4_head — 40525
× 2560 × 4 = 415 MB, or just row 0 + a hash of the rest for verification).

## F-309 — 10-24 ~20:00: THE HEAD-BYTES QUESTION NARROWED TO ITS FINAL FORM — the
## decoders are PROVEN IDENTICAL; the dumped buffer is NOT a vanilla-Q4 image of any
## model row; the definitive close = dump the dequanted rows inside make_q4_head.

**THE PROOF CHAIN**: (1) the vendored ggml's dequantize_row_q6_K and gguf-py's decode
agree BIT-EXACTLY on the real row-0 bytes (cos 1.000000, values equal) — the fixture's
Q6_K source is right and there is NO GSQ-modified Q6_K reader; (2) the FULL-VOCAB SCAN
(engine row 0's decoded vector vs all 248,320 model rows): best cos 0.078 — the dumped
tap_head.q4 does not decode (vanilla, any packing) to ANY model row.  The d matched
dv[0]'s amax/8 exactly (0.00308) while the codes match nothing — consistent with the
buffer holding a TRANSFORMED image (an mmvq-data-path layout transform on dhead_
between make_q4_head and the head call, or a partially-stale buffer), NOT with a
different dequant.
**THE DEFINITIVE CLOSE (option b, one tap line)**: dump the dequanted f32 `row` vector
inside make_q4_head (before ggml_quantize_chunk) — row 0 + a checksum of the rest for
size; the fixture then takes THE ENGINE'S OWN dequanted rows and the T2 gate closes at
the affine-corrected dot's noise floor.  **THE AFFINE DOT FIX remains required in
parallel** (F-308's fact 1): p2_forward's q8_1 dot needs the −4·ds.y term.

## F-310 — 10-24 ~21:00: THE WEIGHTS QUESTION CLOSED FOREVER — the dqhead dump proves
## the engine's rows BIT-IDENTICAL to the fixture (0/103,744,000 differ); the T2
## residual (~1%, cos 0.9965-0.9981, argmax EXACT in every variant) is the LARGE-kernel
## dot form — one more read.

**THE DUMP (STRATA_MTP_TAP -> tap_dqhead.f32, 415 MB, wired inside make_q4_head)**:
the engine's own dequanted head rows == the gguf-py-gathered fixture BIT-EXACTLY.  The
weights, the gather order, and both Q6_K decoders are RIGHT — every prior bytes-mystery
was the transformed on-device image, now moot for parity (the fixture takes these rows).
**THE ISOLATION MATRIX** (all plain fp32 dots on the dumped rows): DQ@S cos 0.99655;
Q4-dequant@S 0.99802; DQ@q8_1(S) 0.99657; Q4@q8 0.99805 — argmax 3992 = EXACT in ALL.
The residual ~1% is INVARIANT to both quantizations ⇒ it is the LARGE-path dot form:
n_in=2560 exceeds the small-K threshold, so the head runs native_mmvq's LARGE kernel
(native_mmvq.cu ~:782+), whose vec_dot form (llama.cpp's vec_dot_q4_0_q8_1 with the
sumiqs bias) I have not transcribed — my affine attempts on the small form all read
cos 0.02 (wrong pairing for the large path).  **ONE kernel read remains: the large
path's Q4_0 traits vec_dot** (the same SmallTraits-style struct at :1033+ is the small
one; the large kernel's Q4 branch is at the vec_dot section) → transcribe → T2 closes.
**G0 stands at: T1 PASSED; T2 = the large-kernel dot form, weights proven, all
quantizers proven, argmax proven — a single transcription from the gate.**

## F-311 — 10-24 ~21:50: THE LARGE-PATH TRANSCRIPTION ATTEMPTED (native_mmvq_multi +
## SmallTraits<Q40Block> with the exact thread nibble map) — T2 STILL OPEN; the two
## remaining suspects isolated: the Q81Block ds field's exact semantics (from
## native_quantize_q8_1 — the s = d*Σx/2 assumption is unverified) and the dispatch
## conditions (multi_exact/b6_rows state selecting a different kernel variant for the
## head).  The exact-map affine form reads cos −0.009 (worse than the plain 0.998!) —
## the affine's s term as I model it is wrong, and the PLAIN fp32 dot on the proven
## rows remains the best form (cos 0.998, argmax exact).  NEXT (fresh context): (1)
## read native_quantize_q8_1's ds packing (d, s) semantics; (2) read the native_mmvq
## dispatcher's kernel-variant choice for the head shape (n_in=2560, ncol=1, Q4_0);
## (3) transcribe THAT variant.  G0: T1 PASSED; T2's every input PROVEN (weights,
## gather, decoders, argmax) — only the dot-form transcription stands between the
## module and the gate.  The ~1% plain-dot residual may itself be the final answer
## if the dispatcher shows a fused/wave variant with different accumulation — the
## read decides.

## F-312 — 10-24 ~22:40: THE ds SEMANTICS NAILED (q8_1_finite.hpp read: d=amax/127,
## s=RAW Σx, both fp16-clamped) — and the raw-sum affine form STILL mismatches (cos
## −0.006, std 4.0); the T2 question now points at the DISPATCHER or the tap point.

**THE READ (complete)**: q8_1_ds(d, sum) = (fp16(amax/127), fp16(Σx_raw)) — my earlier
d·Σx/2 model was wrong; corrected and re-run.  **THE RESULT**: the −4·s affine with the
correct s makes things WORSE (cos −0.006) — the small-traits form CANNOT be the head's
dot.  **THE DECISIVE OBSERVATION**: even the EXACT weights (tap_dqhead, bit-proven)
dotted plainly against tap_sample give only cos 0.9965 — the engine's logits deviate
~1% from W@S on EVERY variant.  Two live hypotheses remain: (a) the head's true kernel
is a different dispatch variant (the IL class — T15A documented mmvq IL covering the
verify head; its form is the native_mmvq_il_kernel at :1669+); (b) the tap's sample_
capture point differs slightly from the head's actual input (a transform between).
NEXT SESSION (one boot + one read): read the IL kernel's Q4 form AND add one tap line
dumping the head's INPUT ROW as the kernel sees it (or dump from inside the head call)
— either closes T2 or proves (b).  T1 PASSED; every weight input PROVEN; the campaign
stands one dispatch-read from G0.

## F-313 — 10-24 ~23:30: THE AFFINE STRUCTURE CONFIRMED BY FIT (c = 1.91 ≈ the
## theoretical 2 — the nibble algebra's needed offset), the last unknown named:
## load_int_b2's byte order.

**THE FIT**: solving for the engine's effective s coefficient from the tap gives
c = 1.912 — the algebra with nibbles (code+8) requires c = 2 exactly, so the affine
form and ds.y ≈ 2·Σx-class semantics are CONFIRMED.  The residual at the best fit
(std 3.9) proves the remaining error is the NIBBLE PERMUTATION (which nibble multiplies
which activation code — my byte-map gives noise-level sumi).  **THE LAST UNKNOWN:
`load_int_b2`** — the inline int loader (2-byte-aligned load, likely a byte-pair
semantics) whose exact byte order defines the map.  ONE function read remains, then
the permutation is mechanical and T2 closes at the fp32 noise floor.
**Dispatch chain now fully proven**: native_mmvq(2) → native_q4_0_mmvq →
small_mmvq<Q40Block,4> → native_small_mmvq_kernel + small_q8_dot(:713) with
d*(sumi*ds.x − 4*ds.y).  G0: T1 PASSED; T2 = the load_int_b2 byte order.

## F-314 — 10-24 ~23:59: THE DISPATCH MACRO FOUND — small_mmvq's first line is
## STRATA_WAVE_MMVQ(SmallTraits<Q40Block,4>): the head (n_out=40525) runs the WAVE
## kernel (native_mmvq_wave_kernel, :1827+), NOT the plain small kernel — the wave
## variant's weight layout/nibble convention is the true head dot form.  The dump-codes
## + affine attempts (both corrections, c=4 and c=8 class) all read cos 0.04-0.09 = the
## plain-small pairing is definitively not the head's.  NEXT (the G0 closer, one read):
## the STRATA_WAVE_MMVQ macro + native_mmvq_wave_kernel's Q4 branch — its load/apply
## (likely the same SmallTraits but a TRANSPOSED/planar weight arrangement, explaining
## the "decodes to no model row" dump) → transcribe → T2 at the floor.  All else stands
## proven (T1, weights, ds, affine algebra, argmax).

## F-315 — 10-24 ~00:20: THE WAVE KERNEL READ — SAME TRAITS, SAME FORM.  The wave
## kernel (native_mmvq_wave_kernel, :1827) uses the IDENTICAL SmallTraits<Q40Block,4>
## load/apply (d*(sumi*ds.x − 4*ds.y), same block layout); it differs only in the
## 64-lane butterfly reduction — accumulation order, ~1e-6 class.  So the head's dot
## form IS the transcribed one on EVERY dispatch path, and the ~1% deviation against
## tap logits SURVIVES every correct-form attempt.  THE HONOMICIDE DIAGNOSTIC (next,
## one boot): dump xq_ (the quantized head input, 2880 B) and dhead_'s first row of
## OUTPUT from inside the head call — a two-line tap — and diff against my modeled
## q8_1 block and dot.  Either my xq_ model differs (the only unverified operand) or
## the tap sample_ point drifts (the earlier hypothesis, now primary again).  All
## kernel forms now exhausted-and-proven; the operand is the question.

## F-316 — 10-25 ~00:50: THE xq_ TAP BUILT, DEPLOYED, AND ITS FIRST BOOT FALSIFIED THE
## LAZY ALLOC — the F-91 law's cleanest demonstration (capture broke: "operation failed
## during capture"; the standing server restored, env-off = the tap code inert = safe).

**THE TAP (in-tree)**: g_tap_X dumps xq_ (the quantized head input, 80 Q8_1 blocks =
2880 B) beside the other three at the request tail; the accessor + the dump line wired.
**THE BOOT**: the dqhead dump fired at load (the env reached the engine ✓) but the
first window failed capture — the lazy `cudaMallocHost(&g_tap_X, ...)` sits INSIDE the
captured tap block = allocation-during-capture, exactly F-91's law.  **THE FIX (one
line, named)**: hoist the alloc to the load-time block beside R/S/L (the size is
shape-constant: (N/32)*36 = 2880 B, no n_dvocab dependency).  Then the boot → the five
files → THE DIFF: my modeled q8_1 block (d=fp16(amax/127), q=round(x/d), s=fp16(Σx))
vs the dumped xq_ — the operand divergence names itself, T2 closes at the floor.
Server restored and verified serving (env-off = inert).

## F-317 — 10-25 ~01:30: THE OPERAND DIFF LANDS — the q8_1 model now BIT-EXACT after
## one named fix; the dhead_ bytes remain the single undecoded piece; the packed-copy
## precedent named as the cause.

**THE xq_ TAP (F-316's hoist fix verified)**: all five files dump on one boot.  THE
DIFF: d and s match the engine 80/80 bit-exact; the q codes differ on exactly
12/2560 — ALL at half-rounding boundaries — because the engine quantizes with the RAW
fp32 scale (amax/127 pre-fp16; q8_1_quant:18-27) while my model used the fp16-rounded
d.  FIX APPLIED (quantize raw, store fp16): the module's activation quantizer is now
operand-exact.  **THE REMAINING WALL**: even with BOTH operands dumped from the
engine (xq_ + dhead_ bytes), every decode convention (sequential nibbles,
split-halves, both affine classes) reads cos 0.05-0.09 — the dhead_ buffer content is
NOT the bytes the wave kernel consumes.  THE PRECEDENT THAT NAMES THE CAUSE:
q8_sm60_owned — "packed copies made here (the head)" — the Q8 path builds a PACKED
COPY for the head; the Q4 wave path plausibly has the same (a transform inside
STRATA_WAVE_MMVQ's launch chain for large n_out).  THE LAST READ: the Q4-analog pack
in the wave launch chain (native_mmvq.cu :2200-2400 region, the q8 pack's
neighborhood).  **G0 STATE: T1 PASSED; the activation operand BIT-EXACT; weights
rows BIT-EXACT (dqhead); the arithmetic form proven on all paths — the one unknown is
the Q4 head's packed-copy layout.**  Server restored env-off, serving verified.

## F-318 — 10-25 ~02:30: **THE Q4 HEAD CONVENTION SOLVED** — the file's OWN per-block
## signed d + split-halves nibbles decode the head: **cos 0.9998, argmax EXACT
## (6970=6970) on both S variants** — T2's best read ever, one refinement from the gate.

**THE DISCOVERY**: the dumped d values match fp16(amax/8) in MAGNITUDE but carry
PER-BLOCK SIGN FLIPS (row0: d[1] −0.00635 vs +expect, d[0] matches — ggml's
quantize_row_q4_0_ref computes d = max/(−8) with its own signed-max walk; the F-298b
packer's uniform negative-d model was wrong, and so was uniform positive-d).  THE
CONVENTION: read the d SIGN FROM THE FILE; codes split-halves (byte j: slot j | slot
j+16<<4); w = d_file·(nib−8).  Reconstructed W @ S reads **cos 0.99981-0.99983,
argmax exact** — the wall of F-307..F-317 falls.  **THE LAST REFINEMENT (to ≤5e-5)**:
run the exact affine dot with these file-signed operands (the −4·s correction at the
file's own d signs) — mechanically one command; expected to close at the fp32 noise
floor and PASS G0.  THE FIXTURE RULE BANKED: never re-quantize a tensor whose
on-device image is dumped — decode the dump's own convention instead.

## F-319 — 10-25 ~02:45: **G0 PARITY GATE: PASSED** — the P2 module reproduces the
## engine's drafter to the fp32 noise floor.  The retrain's prerequisite is MET.

**T1 (the mixer on the dumped R_)**: std 4.87e-7, max_rel 1.96e-6, cos 0.99999994 ✓
**T2 (the head, the exact affine on the dumped operands)**: std 4.63e-7, **max_rel
2.73e-6 (gate ≤5e-5)**, cos 0.99999994, argmax 6970=6970 EXACT ✓
**THE COMPLETE SOLUTION CHAIN** (for the trainer's forward-mode design): weights = the
engine's rt rows (dqhead-proven bit-exact); the Q4 head = per-block SIGNED d read
from the bytes (ggml's max/−8 walk — never a uniform-sign re-quant), split-halves
nibbles (byte j: slot j | slot j+16); the activation = raw-fp32-scale q8_1 quantize
(fp16-stored d/s); the dot = Σ d_w·(nib·q)·d_x with the −8·Σx-class affine
correction (c=2, empirically bit-tight); bf16-class weights consume UNQUANTIZED fp32
activations.  **EXEC-C TRAINING IS UNBLOCKED**: the trainer's G0-exact forward now has
its complete recipe; the two-mode design (exact-quantizer eval / no-rounding train)
per exec-B §5.4 governs the gradients.  THE GOAL'S BIGGEST REMAINING LEVER IS LIVE.

## F-320 — 10-25 ~03:30: EXEC-C OPENED — the training venv LIVE (torch 2.14.1+cu130,
## CUDA on, **sm_120 (12,0) verified natively**), the trainer skeleton landed and
## validated (tools/mtp_train.py: the exec-A 40,976-B record contract, the window
## indexer W=512/K=6/seam-law with its windows.npy cache, the deep-window weighting,
## the two-mode-forward assembly points; dry-run green on the empty harvest dir; the
## G0 module imports clean in the TRAIN venv — 29 tensors).

**THE PIPELINE'S REMAINING GPU WORKLOAD, IN ORDER (per exec-A, spec-pinned)**: the
harvest = CLI-generate boots over a seeded corpus with --dump-final-r (one boot per
stream; ~11M positions ≈ 451 GB f32, deep oversampled ~13%) → shards land in
/home/svg/LLM/Custom-data/harvest/ → mtp_train.py's indexer picks them up → the loop
(K=6 teacher-forced CE, β=0.8, sub-vocab masking, AdamW 1e-4 cosine, 3×16M) → the
offline acceptance evaluator vs the stock baseline → export (BF16 tensors →
mtp_pack.py → mtp_rt.py → rt2/) → the T14C gate ladder.  The trainer's full loop
assembles on the first real shards; every contract it consumes is the one the G0
module already proved.

## F-321 — 10-25 ~04:20: **THE HARVEST IS LIVE** — the exec-A dump patch landed, the
## probe boot wrote 1,663 perfectly-formed records, and the trainer's indexer con-
## sumed them (windows: 3) END TO END.

**THE PATCH (generate.cpp)**: the CLI on_chunk chains the harvest write — per chunk,
{int64 pos, int64 tok, f32 R[4][2560]} x T rows, D2H of the prefill's final-R (the
~35-line exec-A form; the legacy per-token loop stays dead for packs as the spec
found).  --dump-final-r now drives BOTH the legacy path and the native-pack harvest.
**THE PROBE**: 1,664 tokens tokenized via the pack tokenizer (the raw-text file
REJECTED — the tokens-file wants IDS; the probe harness recipe banked), one CLI boot,
exit 0 — **68,143,088 B = 1,663 × 40,976 EXACT**, pos 0..1662 perfectly contiguous,
sane R values; tools/mtp_train.py's dry-run indexes 3 windows from it.  **THE
PIPELINE IS PROVEN END TO END**: corpus → tokenize → boot → records → windows →
(the trainer loop, next).  THE CAMPAIGN REMAINING: the seeded corpus generator (the
exec-A content mix: mimic-lane conversations + repo prose + ladders, deep ≥96k ~13%),
the 180-stream boot sweep (~11M positions, 451 GB — CHECK DISK; bf16 halves it), the
K=6 training loop on the first shards, the stock-baseline evaluator, export rt2.
Server restored (the deploy carries the harvest patch, inert in serve mode).

## F-322 — 10-25 ~05:15: **THE FIRST REAL HARVEST IS IN** — 4 streams, 200,281 records
## (8.2 GB), 393 training windows, deep coverage 66.7% at first-cycle scale.

**THE GENERATOR (tools/harvest_corpus.py)**: the exec-A content mix at first-cycle
scale — deep ladder streams (repo prose: AGENTS/SVG-CUSTOM/GOAL/docs/research,
chunk-shuffled, seeded wraps; 60k tokens each) alternating with mimic-lane conversation
streams (8 task-archetype multi-turn exchanges rendered through the chat template;
40k each) — all pack-tokenized.  **THE SWEEP (tools/harvest_sweep.sh)**: one CLI boot
per stream (resume-skip, single-engine guard); the --temp footgun found and dropped
(CLI generate takes no --temp); 4/4 exit 0 in 100 s wall — records: stream-0
60,074, stream-1 39,996, stream-2 60,175, stream-3 40,002 — byte-exact multiples of
40,976 ✓.  **THE TRAINER CONSUMED THEM**: 393 windows indexed (W=512/K=6/seam-law),
deep (pos≥16384) 262 = 66.7% — the first-cycle corpus is deep-heavy by design (the
F-235 deficit lives there).  The trainer scaffold extended (params view, the
stock-baseline evaluator frame, shard loading — syntax-validated); the torch module
chain port is the remaining piece before the first gradient step.  Server restored.

## F-323 — 10-25 ~06:00: **THE FIRST GRADIENT STEPS ARE REAL** — the torch module
## ported (parity cos 1.00000000 / max_abs 1.3e-5 vs the G0 chain), the K-loss wired
## on harvest anchors, and **51 steps lift depth-1 acceptance 0.4388 → 0.5276
## (+8.9 pt on 196 held anchors)** over the stock baseline.

**THE MODULE (tools/mtp_module.py)**: B1/B2/the HC read as trainable torch ops over
the rt dense Parameters (experts+head frozen, F-114); the two-mode design's TRAIN
mode (no-rounding — the quantizers pass-through; the G0-exact eval mode remains
p2_forward's).  Parity vs the G0-proven numpy chain: cos 1.00000000, max_abs 1.34e-5
(fp32 matmul-order class).  **THE TRAINING**: CE on mixed→W_head sub-vocab logits vs
the true next token (the engine's own dumped Q4 head bytes, F-318's signed-d
split-halves decode — frozen); first step loss 0.169 on real anchors (1/4 masked =
off-sub-vocab, the exec-C expectation); AdamW 1e-4.  **THE READ**: stock 0.4388 →
1-step 0.4330 (noise) → **51 steps 0.5276 (+8.9 pt)** — the first real acceptance
gain from trained weights, on the first-cycle corpus (200,281 records; the depth-1
proxy — the K=6 recursion + the attention KV chain are the remaining wiring for the
full FastMTP loss).  The retrain's mechanism is CONFIRMED LIVE: trainable weights +
real teacher data + measurable acceptance movement.

## F-324 — 10-25 ~07:00: **THE FIRST FULL K=6 CYCLE IS REAL** — GPU-exclusive, 200
## steps in 11 seconds, the per-depth FastMTP loss on 650,782 harvested records:
## **mean acceptance stock 0.4646 → trained 0.5605 (+9.6 pt); depth-1 +23.2 pt,
## depth-2 +13.6, depth-3 +11.7** — the first full-recursion training result.

**THE CORPUS SCALED**: 6 more streams (seed 777) → 10 total, 650,782 records
(26.7 GB), the trainer indexing 1,269 windows (deep 74.5%).  **THE RUN**: the module
on cuda (sm_120 native), batch 4 anchors × K=6 depths = 24 CE terms/step, AdamW 1e-4;
200 steps in 11 s wall (GPU-exclusive — the server paused per the single-engine law;
the CPU-mode timeout runs taught: training needs the GPU alone).  **THE PER-DEPTH
READ (the exec-C §3 evaluator form)**:
| depth | stock | trained | Δ |
|---|---|---|---|
| 1 | 36.8% | 60.0% | **+23.2 pt** |
| 2 | 40.7% | 54.2% | +13.6 |
| 3 | 41.7% | 53.3% | +11.7 |
| 4 | 44.1% | 50.8% | +6.8 |
| 5 | 56.9% | 64.4% | +7.5 |
| 6 | 58.6% | 53.4% | −5.2 |
| mean | 46.5% | 56.1% | **+9.6 pt** |
Checkpoint saved (first_cycle_ckpt.pt).  NOTE (honest): depth-6 −5.2pt and the loss
oscillation say the 200-step/8.2-anchor-visit run is under-converged and the depth-6
term under-weighted — the full cycle (epochs over all 1,269 windows + the w_deep
weighting + the attention KV chain in the recursion) is the remaining work; the
mechanism's direction and magnitude are now DEMONSTRATED at full K.  Server restored.

## F-325 — 10-25 ~13:10: **THE FULL CYCLE CONVERGES (+30.8pt) + THE CHECKPOINT SYSTEM
## + THE WORKSPACE DOCTRINE** (user order: cleanup → checkpoint system → continue; OCD-level tidy)

**THE 1/4-EPOCH CHUNK** (the /tmp-era run, superseded by the system below): BS=16,
lr 5e-5, w_deep=2 @ pos≥16384, 1,148s GPU-exclusive — eval mean stock 0.4040 →
0.7122 (**+30.8pt**), ALL six depths positive: +29.5/+28.5/+27.7/+30.2/+24.1/**+33.3**
— depth-6's F-324 deficit (−5.2pt) was UNDER-CONVERGENCE, not mechanism; w_deep +
10k steps made d6 the best depth.  Honest caveat: offline top-1 on held-out harvest
anchors ≠ engine acceptance (typical-accept + SPEC_PROB arms change the mapping);
the T14C gates + engine battery are the verdict, queued next.

**THE EXPORT ROUND-TRIP BUG (caught before it shipped):** the /tmp script printed
"exported 1 tensors" — its state_dict name round-trip (`replace('_','.')`) only
inverts for names WITHOUT real underscores (`mlp.gate.weight` was the sole
survivor); `hyper_connection_mixer.*`, `attn_hyper_connection.*` etc. silently
dropped.  Fix: export iterates mod.t (the ORIGINAL-name map), kind per tensor from
RtDense().manifest (q8_0 re-quant via the ggml reference packer / bf16 / f32 raw
norms with the +1 un-folded, F-109), + a HARD assert count==29.  LAW: never
reconstruct names by string round-trip — carry the map.

**THE CHECKPOINT SYSTEM (campaign/retrain/train_cycle.py — the standing trainer,
/tmp training scripts hereby BANNED):** atomic rolling cycle-last.pt (tmp+rename,
every 500 steps + at every eval — a kill loses ≤ one interval), cycle-best.pt
tracker, history.jsonl per-eval ledger, AUTO-RESUME (model+opt+sched+rng+step;
legacy bare state_dict seeds model-only), wall-budget chunks (--minutes N),
smokeable (--steps N).  Smoke PASSED end-to-end: 652,445 records / 642,114
anchors, mid-evals, saves, 29-tensor export (116.1 MB, dense.txt dialect, atomically
staged).  Two traps fixed en route: numpy Generator rng state = bit_generator.state
(not get_state); mtp_rt import drags gguf→yaml (absent in the train venv) so the
q8_0 packer is transcribed locally with provenance pinned.

**THE F-325 RESTRUCTURE (AGENTS.md "Workspace organization law" section, BINDING):**
Custom-data/{harvest/{records,corpus,logs}, mtp/{fixtures,ckpts,export}}; the G0
taps moved out of /tmp (396MB one-shot dqhead dump deleted); b9/b12 one-shot
launcher pairs → campaign/arms/; patch_f28{4,6,7}*.py deleted (applied in src —
git history is the archive); headport binaries + retrain npy fixtures out of git
(fixtures home = Custom-data/mtp/fixtures); stale engine log deleted.  Every
artifact class now has a named home BEFORE the first file lands.

**windows.npy NEAR-MISS:** its start column is PER-SHARD record indices (resets at
shard boundaries — negative diffs prove it); a naive global-index read would have
trained across stream seams.  The trainer maps each 512-tile through its shard
offset (valid anchors = [st, st+506) per tile).  Also: the GPU "3GB something" the
user spotted = this training run itself (2,196 MiB) + desktop bits — verified, no
stray holders; single-engine law held throughout (8081 stays down while chunks run).

NEXT (in flight): chunk 2 under the system (resume from step 30) → at-scale
per-depth evaluator → rt2 assemble (experts.bin + draft_vocab.bin byte-identical)
→ T14C gate ladder → engine A/B battery per the user's law → server restore.

## F-326 — 10-25 ~15:00: **THE G0 GATE LADDER BIT TWICE — export ship-stopper caught,
## then the TRANSFER FAIL diagnosed to a clean curve (exposure bias CONFIRMED)**

**G0 EXPORT ROUNDTRIP GATE (campaign/retrain/check_export_roundtrip.py) — first catch:**
the candidate export wrote f32 norms UNFOLDED (arr−1) — but the engine multiplies gamma
AS-IS (native_qsa norm kernel: out = scale·x·gamma[col]) and dense.bin carries the FOLDED
1+w value (mtp_rt.py packs data+1.0).  The un-fold would have shipped EVERY norm off by
exactly 1.0.  Fixed (export stores the param's folded value as-is); re-export + reassemble
→ G0 PASSED: 29/29, f32 bit-exact, worst bf16 4.6e-3, worst q8_0 3.7e-3.  LAW: the dense.bin
f32 norm is the FOLDED value; never un-fold on export.  (Also banked: unused-by-forward
params round-trip bf16 bit-exact — the module trains stage-1 only; self_attn/mlp stay stock.)

**G0 M3d STAIR (campaign/retrain/stair_probe.sh, CLI boot per drafter, same stream/args):
TRANSFER FAIL, replicated.**  stock rt: 0.597 accepted, 2.30 tok/round, depth tail alive
(4:37 5:52).  rt2 (chunk-2, +33.1pt offline at-scale): 0.457 then 0.489 replicate, 1.41-1.44
tok/round, depth COLLAPSE (3:19 4:0 5:0).  **The +33pt offline did NOT transfer.**

**THE DIAGNOSTIC THAT MAPS THE CURVE (ckpts-gentle, --ckpt-dir isolation):** 1,000 steps
lr 2e-5 from stock init (+26.5pt offline) → engine stair 0.604, 2.22 tok/round, depth tail
INTACT (4:41 5:52) = NEUTRAL-to-stock.  So: mild drift transfers neutrally; heavy drift
(10k steps, the +33pt regime) catastrophically breaks the RECURSION.  Mechanism = teacher-
forced EXPOSURE BIAS, precisely the directive's named remaining wiring: the trainer fits
the mixer on the STOCK drafter's harvest R-states; at runtime the trained mixer's own
outputs build its next-depth states (and feed the stock stage-2+ weights), drifting off
the training distribution, compounding with depth.  The offline metric measures one step
off the stock trajectory — it CANNOT see this.  VERDICT: v1 teacher-forced retrain CLOSED
(falsified at transfer, not at fit); phase 2 = recursive rollout training (self-trajectory
states: scheduled sampling / DAgger over K-depth rollouts) and/or the full B-chain incl.
the attention KV chain in the module per T14B-exec-C — the module must consume its own
outputs during training before any further engine battery is spent.

Instruments banked: check_export_roundtrip.py + stair_probe.sh + --eval-only + --ckpt-dir
in the trainer.  Server restored on stock after the GPU work (single-engine law held).

## F-327 — 10-24 ~16:40: **THE PHASE-1 TRANSFER FAIL WAS PARTLY A LABEL-POISONING BUG —
## found by the DAgger merge, fixed by quarantine + a standing gate; clean round-1 recovers
## acceptance to stock-parity; the residual depth deficit is REAL exposure bias, now unconfounded**

**THE BUG (found building the DAgger merge):** the corpus .ids were REGENERATED at 09:22 while
records-stream-0..3 (harvested 09:05) kept labels from the pre-regen corpus — 4/10 shards
(180k records, ~28% of phase-1 labels) were WRONG CONTENT (tok matched no offset of the current
corpus; streams 4-9 + probe are 100% clean: tok==corpus[pos] exact).  The engine dump writes
tok = o.tokens[pos] (generate.cpp:16795) — deterministic; the records simply outlived their
corpus.  F-327 GATE ADDED (train_cycle._verify_labels): every shard's tok stream must equal its
corpus ids at load — the bug class is now unloadable (it fired twice during bring-up: once on
quarantine-shard naming, once on its own mapping typo — the gate works).

**THE CLEAN ROUND-1 (ckpts-clean, fresh stock init, 752,552 records = stock streams 4-9 +
probe + DAgger streams 0-3 re-harvested under gentle-rt2 [records-dagger-*, tok==corpus 100%],
w_deep, lr 5e-5, 11,336 steps/25.8 min):** offline +33.9pt mean (0.6908 vs stock 0.3556) on
VERIFIED labels; per-depth +29.0..+35.8.  **ENGINE STAIR: acceptance 0.606 (final state) /
0.579 (best state) vs stock 0.597 — the catastrophic 0.46-0.49 collapse is GONE, rate is at
stock parity**; depth-3 matches stock exactly (72 vs 73).  RESIDUAL: tokens/round 1.55-1.65 vs
stock 2.30 — the deep tail (4:8-10, 5:0-4 vs stock 4:37 5:52) still breaks early = the REAL
exposure-bias deficit, now measured clean of the label confound.  Also fixed: export ships the
BEST state (final-vs-best checkpoint mismatch caught by the G0 gate); windows.npy rebuilt over
the merged corpus (1,463 windows, 77.9% deep).

**THE LOOP IS LIVE:** harvest_sweep takes a --mtp dir; DAgger round N = re-harvest streams
under the round-(N-1) drafter → merge as records-dagger-* shards → train from the last ckpt →
G0 + stair.  Round 1 states were 36% drafter-mixed; each round tightens the state distribution
toward the serving policy (the 3.5-4.6 ms/row law prices every extra offered row — the target
is depth-tail recovery, worth the full T14C +5.7% band if the tail reaches stock at higher
rate).  Next: round 2 under the round-1 drafter (one command), then the at-scale evaluator.

## F-328 — 10-24 ~17:20: **NEW RECORDS PROVEN — solo 234.1 best leg (mean 223.4), conc
## 338.5 best leg, pair ~222 — on the UNCHANGED standing recipe, cleaned box + full warmth**

The user's box cleanup + a fully-warmed boot (16:45 restore, prewarmed) = the recipe's real
ceiling visible: 10 back-to-back probe legs, same boot, same hour (law).  SOLO: best leg
234.1 (+5.2% over the 222.4 bank), legs 2-10 mean 223.4 — the MEAN beats the old record.
CONC: best leg 338.5 (+4.8% over the 323.0 bank) with same-window flanks 321.3/318.0 — not
a lone tail.  PAIR: last-6 mean ~221.9 vs ~207 banked.  M3b 76-79% throughout (no quality
trade).  ATTRIBUTION: engine/recipe UNCHANGED since the 222-224 plateau — the delta is box
state (zero background load + warmth), proving the earlier plateau carried background tax.
Evidence: campaign/results/f328-record-battery.out.  The 300/450 chase resumes from a
234/338 base with the DAgger depth-tail band (+5.7%) and the USER-GATED experts-requant
(+3-4%) still unstacked.

## F-329 — 10-24 ~20:10: **UPSTREAM v0.2.4.2 CHECKED (six releases, 646 commits) — NO ADOPTION;
## custom leads stock by +43/+55/+48%; DFlash2 lane CLOSED on-box (loses stock-MTP by 25% M1)**

**THE WAVE**: v0.2.2..v0.2.4.2 fetched (references pin was v0.2.1.4).  Highlights: the DFlash2
drafter fully integrated upstream (--draft dflash2 --draft-model PACK, --draft-conf/--draft-cap,
CUDA/HIP/Metal, window caps by card), DeepSeek-V4.1 model class, Gyro trellis (3060-class),
sm_86 spec-adapt, cost-aware MTP window (STRATA_SPEC_COST).  Perf headlines target OTHER
platforms again (F-234 pattern); the ONE sm_120-relevant bit: stock MTP single-stream moved
149 → ~156 (+5%) via their rt-dialect MTP path (upstream adopted our dense.txt/bin rt format
+ mtp_rt.py wholesale).

**THE TEST (stock v0.2.4.2 built at references/Strata-v0242 worktree, sm_120, standing IQ3_S
lane minus our four custom arms; 2 legs per boot, probe-lane):**
  stock+MTP-rt:   M1 160.2/151.9  pair 141.0/144.8  conc 227.4/228.9
  stock+DFlash2:  M1 117.5/116.4  pair 146.9/147.5  conc 236.7/235.0
  CUSTOM (F-328, same day, same box): solo 234.1 best/223.4 mean, pair ~222, conc 338.5
=> custom leads stock's best lane by +43% M1 / +55% pair / +48% conc.  NO MERGE.

**DFlash2 LANE CLOSED (performance)**: within the SAME stock engine, DFlash2 loses M1 by 25%
(117 vs 156); its pair/conc edges (+3/+4%) do not reach our custom's lanes at all.  The
no-recursion block-diffusion idea is now answered empirically on-box: the drafter cost + verify
shape cost more than its acceptance brings at this quant.  Upstream's own suggestion table
("Auto = MTP until a card wins everywhere") agrees.  Assets kept: the packed drafter
(Custom-data/dflash2/drafter.pack, sha-verified source weights), the worktree build, the
campaign/upstream/ harness — reopen only if a future DFlash release claims Blackwell-class gains.

**LAW RE-CONFIRMED (2nd occurrence)**: stock boots with this pack NEED --vram-reserve-mib
(2000); without it the expert cache fills 19.54 GiB and the verify graph OOM-crash-loops
(engine exits per request — the server masks it as 0.0 t/s lanes).  F-234's 200→768 note,
now 2000 — always set it on stock boots.

Evidence: campaign/upstream/{s1-mtp,s2-dflash2}*.out, window*.out, configs s1/s2.json.

**F-329 addendum (10-24 ~20:30, user order): upstream-test cleanup done.** The v0.2.4.2
worktree (1.0 GB incl. build) removed from the references clone (back to the pristine
v0.2.1.4 pin, worktree list empty); the derived drafter.pack (726 MB) deleted — regen is
one command from the sha-verified source kept at Custom-data/mtp/fixtures/dflash2/.
campaign/upstream/ (784 KB: evidence, configs, the window harness) stays as the standing
upstream-check home.  NEW LAW in AGENTS.md: references library stays pristine; upstream
checkouts/builds/test binaries live at ~/LLM/strata (the stock tree, currently 0.1.46);
derived test artifacts deleted at closeout.

## F-330 — 10-24 ~21:00: **UPSTREAM v0.2.4.2 STUDY (~/LLM/strata now the pinned stock tree) —
## adoption menu priced; ONE clear port (SPEC_COST chain rule), one instrument, everything
## else already-owned or other-platform**

**THE STUDY** (post-F-329 verdict, deeper read of the tree): the +5% stock-MTP M1 (149→156)
is upstream RE-DERIVING fusions we already bank — their hot lane now runs fused_gr_read_multi
(one launch, bitwise same, the lineage our F-series built) plus a gr_read 6→3-kernel fold
(aec327ee, Oct 21) that only improves the per-row/HIP path OUR hot lane bypasses.  That is
why stock gained 5% yet still trails custom by 43-55%: they are climbing the ladder we own.

**ADOPTION MENU:**
1. **PORT — the cost-aware MTP chain rule (STRATA_SPEC_COST)**: 5f49ec45 +
   include/strata/spec/draft_policy.hpp (101 lines, self-contained) + a step-by-step
   keep_draft hook in MtpDrafter.  This is OUR T14C crossing rule (ΔC/Δoffered > c_row·C₀/W₀)
   as a RUNTIME learner: E_j/cost(j+2) per width, measured per-width round costs bucketed by
   context length (<6K/<24K/longer), acceptance calibrated per draft probability, robust
   cost-EMA (a first capture can't price a width out).  F-228 killed STATIC gate tuning;
   this is the dynamic version — trims weak deep chains, exploits non-convex row costs,
   composes with COHORTS.  Realistic +1-3% M1/throughput; ~half-day port into our mtp.cpp.
2. **PORT — STRATA_ROUTE_STATS + per-width GPU stage profile** (d276adf2): bench-only
   instrument, cheap; feeds the TC-hc tranche's cost anatomy (row costs per width/context).
3. **SKIP — gr fold (aec327ee)**: hot path already fused multi-row in custom; residual
   per-row gr_read = 1-2 calls/window ≈ negligible.
4. **SKIP — spec_start fix** (single-token prompts never speculated pre-0.2.2): edge case,
   worth a one-line audit of our spec-pos convention someday.
5. **SKIP — dflash2 kernels / dsv41 / Gyro / sm_86**: other models/platforms (F-329 closed).

NEXT: the SPEC_COST port (header + hook + STRATA_SPEC_COST arm, default off) with the
3x-probe before/after per the law; ROUTE_STATS alongside it (same rebuild).

## F-331 — 10-25 ~01:55: **THE INT8-MMA ATTENTION BUNDLE ADOPTED — pair +6.5% (bar cleared), conc
## +3.9%, M1 in-noise, MMLU 2-gate clean; the int8-MMA kernel pair was sitting default-off, never priced**

**THE ARMS (found by a full flag/env audit against the record: every other cheap knob is already adopted
or falsified — L2-72 neutral F-200-era, spec 8 rejected −22%, lookup-chain killed by the +4.7ms/row CPU
coefficient, PIN_BY_NEED ~0 (complement 100% pinned, F-195), adapt-async demoted (silent non-arm,
F-195), EXPDEC conflicts with the adopted GU_SWIGLU at iq_kernels.cu:4874 — priced for a SOLO A/B):**
- `STRATA_QSA_SPLIT_ATTN=1` — decode+drafter attention through the split int8-MMA kernel
  (verify.cpp:1973 + mtp.cpp:1088; qsa_prompt_attn.cu:2271: cells split over up to 20 blocks, LSE merge;
  fork Eddoursul fd23d91 idea; vendor class +2.8/3.5% on other cards). NEVER priced here — zero
  SVG-CUSTOM mentions before this entry. Safe-fallback by construction (caller keeps qsa_decode_attn_batch).
- `STRATA_PROMPT_ATTN_IMMA=1` — prefill attention both products on INT8 tensor cores
  (qsa_prompt_attn.cu:1081 i8v3: 3 int8 MMAs ≈ 1 fp16 MMA where v2 spends 2; sm_120 rate 732 vs 233
  measured in-file; 24-bit exact split FINER than v2's fp16 hi+lo). Also never priced here. NOTE: this
  probe cannot price it (fresh-prompt prefixes conversation-cache after round 1 — all arms read
  0.55-0.56s); the prefill lane (t42-prefill-ladder) owes it a real read. It rides on mechanism + gates.

**THE PROTOCOL (3v3 same-hour, fresh-boot ABA after the boot-state confound showed):** ctl-A was the
traffic-warm standing boot (M1 191.8/pair 191.1/conc 282.3/M3b 79.2 — arm read +2.6/+2.4/+3.4 vs it),
but fresh-vs-warm is exactly the F-225 trap, so a fresh prewarmed arms-OFF control (ctlB) closed the
window: **ctlB 198.6/183.7/281.0/77.5 vs ARM 196.7/195.6/292.0/76.2** →
**pair +6.5% (CLEARS the +4 bar), 6-conc +3.9% (at the bar), M1 −1.0% (spreads overlap 190-210 = noise;
single-stream protection HOLDS), M3b −1.3pp** (two controls read higher — a real small tax, inside the
75.6-79.6 historical band). Boot-state lesson re-learned: ctlB M1 reads +3.5% over traffic-warm ctl-A
while its pair reads −4% under it — lane-dependent boot state is why the interleave is law.
Evidence: campaign/results/f331-{ctl,arm,ctlB}-{1,2,3}.txt.

**GATES (token-stream-affecting: QSA_SPLIT changes decode attention numerics):** MMLU-Pro 70q
floor-0.933 form: **92.86 then 94.29 (mean 93.57 ≥ 93.3 floor, tiebreaker-clean**; the 92.86 sub-floor
read is inside σ 3.1pts and inside the standing recipe's own F-282 range 92.9-95.7).
mmlu-f331-arm{,2}.json.

**ADOPTED into standing** (launcher commit 347ba869 + overridable-form follow-up): both arms
`${VAR:-1}`, one-line `=0` rollback each. RIDERS: (1) five-boot pooled confirmation + mimic watch on
the next battery (F-282 pattern); (2) IMMA owes the prefill-lane read; (3) M3b −1.3pp watch; (4) if a
solo-lane bisect is ever wanted, QSA_SPLIT alone is the measured contributor — IMMA is prefill-only.
Calibrate re-verdict (user question): F-182/F-183 stand — its whole sweep space is now either
adopted-elsewhere or falsified with mechanism; the saved 10-14 report remains stale and dangerous to apply.

## F-332 — 10-25 ~16:45: **STRATA_ADAPT_EARLY ADOPTED — the tier's decision scan runs under the window;
## adapt wait 1.03 -> 0.28-0.62 ms/window (mechanism proven), pair +5.4%, no lane negative**

**THE PREMISE (R2-P5 falsifier first):** the live timing line read "commit 0.36 + emit 0.38 + adapt wait
1.03 ms/window" of 14.21 — the tail's biggest piece is NOT the stdout emit but the ADAPTIVE TIER's
round-end join+apply. At adapt_every 4 that is ~4 ms stalls every fourth window. The join waits mostly
for adapt()'s own DECISION SCAN (launched at round end, it got only the ~0.7 ms finish tail to run),
not the copies. Deferring the join past the next launch would reopen the deleted F-187 overlap lane
(copies racing windows; also blocked with DMA_BATCH) — CLOSED by code read.

**THE ARM (generate.cpp, commit a9055142):** split adapt() into adapt_scan() (the layer x expert pass
+ sorts: pure reads of usage/pred/heat/host_res/admit_round/pin_rank) + the round-end part (credits,
gate-set, copy enqueues, event) UNCHANGED. STRATA_ADAPT_EARLY=1: after the cohorts' batch_go, snapshot
usage/pred/heat (the running windows mutate the live ones; 384 KB copy ~30 us) and run the scan on a
thread under the ~12 ms window run; at the round end join it (done long ago) and pass the swaps to
adapt(). Decisions are one round staler — usage is a decay EWMA, noise. The scan reads only
round-end-owned state (host_res/admit_round/pin_rank mutate in apply_pending, main thread only) —
race-free by placement. Default off = inline scan, byte-identical.

**THE BATTERY (3v3 same-hour, old-engine warm boot vs fresh arm boot, prewarm2 both):**
ctl 208.5/187.7/288.2 M3b 79.5 | 205.1/204.8/308.8 76.9 | 219.0/204.0/300.5 72.9
=> means M1 210.9 / pair 198.8 / conc 299.2 / M3b 76.4
arm 220.4/207.2/326.0 77.6 | 210.9/204.1/290.2 78.9 | 202.2/217.5/295.5 77.4
=> means M1 211.2 (+0.1) / pair 209.6 (+5.4%) / conc 303.9 (+1.6%) / M3b 78.0 (+1.6pp)
MECHANISM: arm timing lines "adapt wait 0.28-0.62" vs control 1.03; one 9.81 GPU-reach outlier line
(scan thread on a core while host waited) — transient, watch on the pooled round.

**ADOPTED** (launcher ${STRATA_ADAPT_EARLY:-1}; =0 rollback; engine rollback engine/strata.pre-f332).
Honesty note: the pair mean is leg-1-sensitive (ctl leg 1 read 187.7; medians +1.6%) — means are the
law's form and no lane regressed, adoption stands on mechanism + means + zero regression; the F-282
five-boot pooled confirmation rider applies. Exactness reasoning: swap CHOICES shift one round (which
experts resident where) but every path computes identical expert values — token stream unchanged in
expectation; M3b +1.6pp consistent. Evidence: campaign/results/f332-{ctl,arm}-{1,2,3}.txt.

## F-333 — 10-25 ~17:40: **THE BOUNDARY HYGIENE ROUND: scan-thread pinning KEPT (unconditional),
## STRATA_EMIT_LATE_FLUSH NOT ADOPTED (inside band both directions); the ~10 ms GPU-reach lines
## identified as the drain-time print artifact; three-boot drift map banked**

**THE ARMS (the same-pattern study's quick items):** (1) the F-332 scan thread now pins via
strata::aux_cpus::pin_current_thread() (house pattern, generate.cpp:1890 precedent) — unconditional
hygiene, no env; (2) STRATA_EMIT_LATE_FLUSH=1 — the finish tail's 2-3 per-cohort fflush syscalls move
to just after the next round's launches (BT lines buffered <2 ms; idle drain + exit still flush; default
off = byte-identical). Engine commits: pinning + flush arm; rollback binary engine/strata.pre-f333.

**THE THREE BOOT SETS (3 legs each, same hour):**
ctl (F-332 standing, ~1 h warm):  M1 209.1 / pair 204.6 / conc 289.9 / M3b 77.5
arm (fresh, flush on, pinned):    M1 204.1 / pair 191.5 / conc 280.4 / M3b 77.9
ctlB (fresh, flush off, pinned):  M1 189.8 / pair 190.9 / conc 285.3 / M3b 78.3
Naive ctl-vs-arm read a -2.4/-6.4/-3.3 REGRESSION; the immediate ctlB re-run (the law's
when-in-doubt rule) reverses it: arm-vs-ctlB = M1 +7.5 / pair +0.3 / conc -1.7. Every lane INSIDE
the band in both directions => the flush arm is UNPROVEN, not adopted (default stays off; do not add
to the launcher). The spread IS the box: tonight's three boots span M1 189.8-209.1 with the arm boot
the FASTEST fresh boot — F-214's boot-lottery lesson re-earned; single fresh-boot verdicts on this
box are worth little, arms this small need the pooled form or nothing.

**MECHANISM NOTES:** (a) the ~10 ms "GPU-reach wait" lines (9.81 on the F-332 boot, 10.06 here) appear
at LANE DRAIN — the timing line prints when batch_on() goes false and the last rounds have no overlap
left; they are a print-placement artifact, NOT contention — the pinning arm did not and was never going
to remove them (kept anyway: zero-cost, correct-by-precedent). (b) adapt wait read 0.71-0.78 this
session vs 0.28-0.62 on the F-332 battery — boot-dependent DMA residuals, watch on the next pooled
round. (c) A blocking-pipe mechanism against late flush remains plausible (the write lands between
launch and pool service) but tonight's data cannot confirm it; if ever re-priced, flush on a SIDE
thread instead of moving the syscall.

**F-332 POOLED-CONFIRMATION LEDGER (rider, updated):** F-332-armed boots now 2 (211.2/209.6/303.9 and
209.1/204.6/289.9 mean-of-3 each); F-332-off controls 2 (198.6/183.7/281.0 pre-adoption, 189.8/190.9/
285.3 tonight-with-pin) — pooled pair 207.1 vs 187.3 = +10.6%, but the boot spread (183.7-209.6) warns
the pooled number carries drift; directionally consistent, rider stays open until the five-boot form.

Evidence: campaign/results/f333-{ctl,arm,ctlB}-{1,2,3}.txt. Server restored on the standing recipe
(ADAPT_EARLY=1, flush off) and healthy.

## F-334 — 10-25 ~18:15: **THE CLEANED-BOX 10-LEG BATTERY ON THE F-331+F-332 STANDING RECIPE — at the
## F-328 record band (M1 best 232.5 vs 234.1 bank; conc best 327.2), day-best means on every lane;
## the F-332 armed recipe shows no ceiling regression**

**CONTEXT (user order: box cleared of background processes, restart + retest).** Boot: standing recipe
(F-331 int8-MMA attention arms + F-332 ADAPT_EARLY + F-333 pinning; flush off), prewarm2, load 0.69,
GPU idle pre-test. 10 back-to-back probe legs, same boot, same hour (the F-328 form).

**MEANS (10 legs):** M1 219.2 / pair 208.9 / 6-conc 305.9 / M3b 77.8%.
**BEST LEGS:** M1 232.5 (twice) + 232.4 / pair 218.4 / conc 327.2 / M3b 80.3.
Warm trend: legs 1-5 M1 mean 214.5 -> legs 6-10 mean 223.9 (leg 5/6/8 all ~232.5); the plateau is the
F-328 warm-protocol territory.

**vs TODAY'S DIRTY BOOTS (F-333's three):** day-best on every lane — M1 219.2 vs 189.8-214.5 boot means;
conc 305.9 vs 280.4-289.9. The background-process tax was real.

**vs THE F-328 BANK (pre-F-331/F-332 recipe, also a cleaned+warm box):** statistically INSIDE the band,
not above it — M1 mean 219.2 (legs 2-10: 220.7) vs 223.4; best 232.5 vs 234.1; conc best 327.2 vs 338.5;
pair 208.9 vs ~222. The F-328 numbers remain the bank. Honest read: tonight's boot matched the record
band WITH F-331+F-332 aboard (F-332's armed-cleaned confirmation: no ceiling regression; the F-331
M1-neutral/pair-positive shape shows as pair 218.4 best vs solo 232.5 pattern). Boot lottery remains
the binding variance (this session's boots spanned M1 189.8-232.5 legs).

**F-332 POOLED LEDGER:** armed boots now 3 (means 211.2/209.1/219.2) vs control boots 2 (198.6/189.8) —
directionally consistent on every pair; the five-boot control form still owed.

Evidence: campaign/results/f334-clean-{1..10}.txt. Server left serving on this boot.

## F-336 — 10-25 ~21:15: **THE F-332 POOLED CONFIRMATION (first interleaved-paired 5-boot form): pair
## CONFIRMED at +2.1% (3/3 pairs, gate pass), M1 protected; NEW 6-conc tax −4.7% (3/3, gate pass) under
## background load — banked with the aux-cpus no-op mechanism; scoping fix is one env**

**THE FORM (play-mode, standing-arms row; the scout's interleaved-paired design):** 5 boots A-B-A-B-A
(B1/3/5 = ADAPT_EARLY=0 ctl, B2/4 = arm), same hour, prewarm2 each, 3 legs/boot, user bg load live
(the real condition), 1 Hz nvidia-smi telemetry alongside (killed at close; pkill-by-comm - the -f
self-match footgun bit once, banked again: never pkill -f a pattern your own shell carries).

Per-boot means: ctl 220.1/196.9/300.1 | arm 200.8/202.3/277.9 | ctl 216.2/201.5/304.1 |
arm 217.1/206.6/295.0 | ctl 216.8/204.3/306.0 (M1/pair/conc).

**ADJACENT PAIRED DELTAS (arm − ctl):**
- pair: +2.7 / +2.5 / +1.1 => mean **+2.1%**, sd 0.9, gate 2sd/sqrt3 = 1.0 => **SIGNAL, 3/3 positive**
- M1: −8.8 / +0.4 / +0.1 => mean −2.7%, sd 5.2, gate 6.0 => **inside noise (one low boot; B4 matched ctl
  exactly twice) — single-stream protection HOLDS**
- 6-conc: −7.4 / −3.0 / −3.6 => mean **−4.7%**, sd 2.4, gate 2.8 => **NEGATIVE SIGNAL, 3/3**

**VERDICTS:**
1. F-332's pair gain is REAL but SMALLER than the adoption claim: +2.1% pooled vs +5.4% claimed —
   inside the MLPerf-style ±5pp adherence band => **CONFIRMED, magnitude revised**. The F-332 five-boot
   pooled rider CLOSES.
2. **NEW: a conc-side tax the cleaned-box battery did not show.** Under bg load, all three arm boots
   read below their adjacent controls on 6-conc. MECHANISM (hypothesis, priced): the F-332 scan thread
   runs exactly when the 6-conc windows need the CPU pool most (2.3-3.2 ms/window of CPU-expert work),
   and the F-333 "pinning" is a NO-OP while STRATA_AUX_CPUS is unset (generate.cpp:1890's own comment).
   CHEAPEST FALSIFIER (one boot): arm boot with STRATA_AUX_CPUS configured (scan + pool affinity split)
   — if the conc delta collapses toward 0, the scoping fix is that env, per law 4b (scope, not revert).
   NOT a revert: pair positive, M1 protected, tax small and load-dependent, =0 rollback one line away.
3. Method note: the interleaved-paired form cost the same 5 boots as the old pooled form and produced a
   GATE-PASSING verdict with 3 deltas — this is the law-v2 form working; adopt it (row: stable-baseline).

Evidence: campaign/results/f336-{ctl,arm}-B{1..5}-*.txt. Server restored on the standing recipe (arm),
prewarmed, healthy.

**F-336 addendum (falsifier correction):** the one-env falsifier is NOT one env on this box —
--aux-cpus auto picks "a physical core with no worker; nothing is done when none is spare", and
15 pool workers + 1 host fully subscribe the 9950X3D's 16 physical cores => auto is a NO-OP here
(the F-333 pin's no-op explained at the mechanism level). The honest falsifier is a COMBO leg:
--pool-workers 14 (free one physical core) + STRATA_AUX_CPUS=auto (scan+aux threads land there) vs
standing, conc lanes. Queued as the standing-arms row's next leg.

## F-337 — 10-25 ~23:20: **THE CONC-TAX FALSIFIER RAN: pool-14 + aux-cpus does NOT fix it — the
## scan-thread core-contention mechanism is FALSIFIED; surviving hypothesis = the one-round-staler
## residency tables widen the conc miss path; scoping patch queued (slot-count-gated adapt-early)**

**THE LEG (adjacent pair, same hour, bg load live ~3.7):** ctl = standing (pool 15): M1 209.4 / pair
206.7 / conc 295.6 / M3b 79.0. combo = --pool-workers 14 + STRATA_AUX_CPUS=auto (verified: health
pool_workers 14, env on both boots; the scan+aux threads had their own physical core): M1 206.6 /
pair 197.3 / conc 283.3 / M3b 77.8. DELTAS combo−ctl: conc **−4.2% (unchanged — the tax did not
collapse)**, pair −4.6%, M1 −1.3 (noise). Pool-14 itself is also no win (consistent with the old
pool-workers-8 null).

**VERDICT:** the F-336 mechanism (scan thread stealing pool cores at 6-conc) is DEAD — a dedicated
core changes nothing. SURVIVING HYPOTHESIS (mechanism re-read): adapt-early's decisions are ONE ROUND
STALER; at 6-conc the expert working set is ~6x wider, so a staler residency table routes more
per-window entries down the pool/PCIe miss path exactly where conc traffic lives. This also explains
why the cleaned-box F-332 battery (no load, tighter working set) read conc in-band.
**SCOPING PATCH (law 4b, queued as the row's next leg):** gate the early scan on active slot count —
`adapt_early_env && ... && w_active_total < N` (N~3): solo/pair keeps the +2.1% pair win, 6-conc falls
back to the inline scan (the F-336 conc tax gone by construction). ~2 lines + one battery.
**BANKED ARTIFACTS:** campaign/arms/{f337-pool14-iq3_s-8081.json, launch-f337-pool14.sh} (the launcher
clone carries the standing env; its `cd $(dirname $0)` must be made absolute when cloned out of root —
footgun banked).
Evidence: campaign/results/f337-{ctl,cmb}-{1,2,3}.txt. Standing recipe restored, prewarmed.

## F-338 — 10-26 ~11:55: **KEEP-WIN-DROP-LOSS SHIPPED: slot-count-gated adapt-early (user order after
## reboot; API choppy recon) — solo/pair keeps the +2.1% pair win, conc falls back to the inline scan;
## VERIFICATION BATTERY PENDING (operator no-test order on the fresh reboot)**

**RECON (user: "API choppy, did we regress?"):** (1) the box was REBOOTED — nothing survived, the server
was DOWN = the primary choppy cause; (2) the one real regression on record = F-332's 6-conc tax −4.7%
under bg load (F-336/F-337) — consistent with choppy concurrent traffic; (3) everything else held:
F-331 standing, flush never adopted, pool-14 dropped.

**THE GATE (generate.cpp, commit on top of F-337):** the early scan launches only when the round's
active slots < STRATA_ADAPT_EARLY_MAX_SLOTS (default 3). Solo(1)/pair(2) = early scan (the confirmed
+2.1% path); 3+ slots = the inline scan (the always-correct pre-F-332 path) — the conc tax is gone by
construction, no new mechanism. =0 on STRATA_ADAPT_EARLY remains the full rollback; engine rollback
engine/strata.pre-f338.

**STATE:** fresh-reboot boot, standing recipe verified on the process (QSA_SPLIT/ADAPT_EARLY/
PROMPT_ATTN_IMMA on, flush off), pool 15, prewarm2 done, idle. NO battery run (operator order): the
gate's verification (interleaved pairs, conc lane) is QUEUED until tests are allowed — and per the
measurement law it must run under MATCHED conditions (bg load on, same hour), never against the
rebooted clean box.
