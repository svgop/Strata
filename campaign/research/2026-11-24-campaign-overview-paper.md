# STATE OF THE RECORD — the Strata custom campaign, 2026-11-24

Scope: everything the record carries for the svg fork (`/home/svg/LLM/strata-custom-2`, HEAD
`66a0c6beb`), with the last week's work (the quality axis, the PolyStrata/upstream recon, the two
adjudications, the architecture decompositions) given in full. Every number names its owner.
External numbers (PolyStrata, upstream, the anonymous 16 GB laptop rig) are *theirs* — hardware we
do not have, several one-run-each — and are used only as direction and re-pricing, never as claims.

---

## 0. Abstract

The campaign serves Qwen3.8-Flash-Next (IQ3_S) on one RTX 5090 (sm_120, CUDA 13.4) + 184 GiB RAM
and holds, on the record: solo M1 ~220-224 t/s banked means (record legs 228.7-240.0), 6-concurrent
~307-312 aggregate (peaks 316-318), MMLU-Pro gate floor 93.3 with prior readings 94.3/92.9 — all
against stock 0.2.1 baselines of 147/235.5/95.0-era (M1 +~50% over the original stock bank). The
wall is stated as one equation — *host-stage exposure × (1 − rows in flight)* — and every remaining
lever is priced against it. The last week added the missing second axis: the F-450 Pareto rule
(quality ≥ 0 AND perf ≥ 0, one strictly > 0) made quality a first-class battery, and the persona
arc delivered the first Tier-A quality adoption under it (phase-scoped register add, MMLU tied to
stock, refusals held). Externally, the PolyStrata fork and upstream 0.2.2..0.2.20 were adjudicated
against our banked evidence: four items survived (one proven defect, one env-arm cluster, one
measurement lane, two small ports), nine were falsified or dropped with mechanisms — most killed by
falsifiers we had already run, at zero boot cost. What is not achieved is equally precise: the
+10-25% rows-class levers are priced and armed but not cashed; the conc lane past ~312 is unbuilt;
the drafter's acceptance ceiling and the coverage-axis correction are open questions; ten commits
sit unpushed.

---

## 1. The mission and the acceptance rule

The standing objective is the F-450 operator directive (2026-11-23), now codified in
`campaign/standing-line.md` as the acceptance rule for *every* change:

    quality(delta) >= 0  AND  perf(delta) >= 0,   at least one strictly > 0.

A trade on either axis is not a candidate; it is an operator decision with the measured trade on
the table. Two batteries bind, same build pair, engine order reversed: a fixed-prompt quality
suite (greedy AND sampled-with-fixed-seed; greedy under a determinism-pinned config must be
byte-identical or the divergence is a defect first) and a perf battery (TTFT 4K/32K, median decode
over full-length runs only, cold/warm never mixed). The determinism canary is `shadow_misses`
(`include/strata/core/expert_source.hpp:661`): two runs of the same prompt with different
shadow_misses carry different quantization mixes and are not comparable. The older campaign laws
(interleave-settles, +4%/−4% adopt bar, binary ladder, M3b canary, MMLU 93.3 gate for
token-stream-affecting changes) continue to bind; F-450 does not replace them, it adds the
quality axis they lacked.

## 2. The method, as practiced

The doctrine in `AGENTS.md` is not decoration; this window ran on it. The loop — recon →
hypothesis with mechanism → cheapest falsifier → instrument → interleave → verdict → journal,
commit, push — produced its characteristic signature in the adjudications: of the thirteen external
claims examined, nine died without a single boot, five of them killed by banked evidence we
already owned (§4). The negative results are banked with mechanisms and re-open clauses
(`campaign/research/levers-2026-11-20-evening.md` §§8-11), and three of my own pricings were
retracted on the record when the data contradicted them (§5.3). That retraction habit is the
method working, not failing: a wrong number that survives the record gets re-purchased at full
price by the next agent.

## 3. What has been achieved

### 3.1 The performance axis

Standing recipe (all standing, golden-identical where noted): IQ3_S on the 5090, 9900-slot VRAM
cache (18.78 GiB) + 37.86 GiB pinned complement, adaptive tier (every=4, swaps=96, decay 0.85,
PINCAP 48, MIN_AGE 2, DORMANT 64, PREDICT_CREDIT 1.0), ADAPT_EARLY, ROUTE_TAIL_SKIP=7 (T63),
VERIFY_DEVICE_PLAN=1, `--pcie-frac 0.35 --pcie-mode dma`, spec 6 MTP, KV int8, parallel:6, Q2_0
shadow armed, POOL_DIET=1, PROMPT_ATTN_IMMA (launcher:142).

Tracker banks (`campaign/tracker/lanes.csv` — only the standing earns points):

| Date | Lane | Mean | Record leg | Source |
|------|------|------|-----------|--------|
| 2026-11-09 | solo | 223.6 | 233.1 (peak 238-239) | GSQ record-class restored |
| 2026-11-09 | 6-conc | 307.6 | 316-318 peaks | same bank |
| 2026-11-10 | solo | post-reboot warm | 240.0 peak | isolation stack live (isolcpus/nohz_full 8-15+24-31, clocks pinned) |
| 2026-11-20 | solo | 220.6 | 228.7 (boot 5 leg 3) | F-405/F-408 quiet-window bank |

Against the AGENTS.md stock-era baselines (M1 147 stock → 198.5 first custom bank, +35.0%;
6-conc 187.6 stock → 312.4 banked, +66.4%), the current standing holds and exceeds every banked
lane. The kernels are not the lever: F-411 measured them at 95-98% of the stream anchor.

The wall, stated precisely (dossier §0, same-boot F-414 data): solo window ~13.6 ms at load 2-3;
GPU-reach idle 8.79 ms at 3.10 rows vs 0.51 ms at 4.88 rows; CPU experts 2.85-3.27 ms; adapt wait
0.76 ms quiet / 1.19 ms under load; emit 0.35-0.61 ms. **wall = host-stage exposure × (1 − rows in
flight).** Every high-value lever either adds rows (depth, acceptance), shrinks a host stage (pool
granularity, swap traffic), or moves a host stage under the GPU window (lookahead). The conc lane
is the same wall at 4.68 rows — its anatomy shows reach idle 1.33 + CPU experts 2.98 + adapt 1.19.

### 3.2 The quality axis (the week's principal achievement)

F-450 turned quality from an adjective into a battery. Under it, the persona arc
(`campaign/research/2026-11-23-vector-fresh-eyes-review.md`, `2026-11-26-vector-v2-research-and-fix.md`,
commits 0bd36187e → ba13d1acb) delivered the first quality-axis standing adoption:

- **The register moves, facts hold.** Deployed, same-window: assertive 13/3 p=0.011, serious 13/1
  p=0.0009, direct 11/3 p=0.029, facts 1.00/1.00, judge controls pass; refusal battery ON 0/10,
  OFF 10/10 (the refusal-r ablation survives composition). Serious survives the length confound
  (14/0 at |dWords| ≤ 12).
- **The dose mechanism is understood, not tuned.** The 0.45 over-dose was a *domain error*, not a
  taste call: hyper-connections broadcast every block output to 4 streams and the read-out is their
  sum, but each block RMS-normalises each stream independently before the gated mix — so a fixed
  add's effective share follows the HARMONIC mean of stream norms, not the arithmetic one the
  build originally calibrated on (1.59× nominal at 0.45, 1.06× at 0.3). The build now calibrates
  in the block's read domain and records the dose domain in provenance (7e6c2878a).
- **The cost is a dose-response, not a flat tax.** Interleaved per-request ON/OFF MMLU in one boot
  (`campaign/mmlu-pro-interleave.py`): dose 0.45 ON 63/70 vs OFF 67/70 (McNemar p=0.219); dose 0.3
  ON 65/70 vs OFF 66/70 (p=1.000). OFF 67/70 is the best standing reading on record.
- **Phase-scoping removed the where/when damage** (ba13d1acb): the add now lands on the answer
  phase only (gate closed through prefill, opened at a generated `</think>`), because the contrast
  is elicited from response tokens and the unperturbed question encoding and deliberation are
  worth keeping. Parity harness caught two kernel bugs before deployment (a device-int write
  targeting the host pointer field; a compound literal's address under nvcc).
- **The two-table engine** (0bd36187e) made the composition possible at all: table 1 (persona add)
  + table 2 (refusal-r project) with per-table modes, a session-digest `cvec2` object, and a
  Layer-2 boot gate that verifies both tables asks-vs-armed. The gate's eleven new tests caught
  two real crashes before any boot relied on it.
- **The SYCL gap is declared, not stubbed**: the SYCL backend parses no `*2` flags, so a two-table
  config is refused at argument parse — a boot that believes it ablates refusal and does not is
  the exact failure a stub would have shipped.

Live-engine quality telemetry (ours): prompt reuse mean 93.2% across 4,820 events; 193 full
re-reads and 266 sub-50% events — the conversation-cache lane's measured target (§6, item 6).

### 3.3 Architecture and reliability

- **The god-module is half decomposed** (F-432..F-436, one variable per leg): plan bookkeeping,
  admission checks, plan emission, the CPU-miss pipeline, and ADAPT_FETCH swap-admission are
  extracted from `plan_build`; each cut certified against the pre-session baseline (V1) with the
  failure taxonomy pinned (V2).
- **Golden Masters own the seams**: `expert_pool_dispatch_multi` executes under test with the
  branch matrix completed (no-plan arm, onepass arm, single-token dispatch); the arena boot organ
  has its own Golden Master (F-438/F-440/F-446).
- **The twice-earned UB class is now a build failure** (F-448), and `read_direct` deduplicated to
  one window builder across platform halves (F-445).
- **The store is closed** (66a0c6beb): 20 GB of ~/LLM (ten parallel trees) and 789 GB of DATA
  audited by closure — every deletion checked against configs, launchers, tools before it happened
  (law 13). Superseded vectors purged (11 → 8 files, every survivor with a live consumer or a
  named role), stale rollback binaries retired, lab scratch reclaimed. The audit found the
  fragmentation's root: the trainer's RECORDS_GLOB resolved to ZERO shards because two data homes
  had diverged — now one home, one references library.

### 3.4 External intel, adjudicated (PolyStrata + upstream 0.2.2..0.2.20)

Three recon sweeps and two independent adjudication subagents against the pristine clone
(`data/references/PolyStrata`; merge-base 90767252f = 0.2.1; their side 7,361 commits to our 654;
release notes say 0.2.7 but the tree carries 0.2.2..0.2.20 content). Their Qwen3.8 lane is the
upstream engine unchanged — not our competitor; their engine work (poly, 13 models) and hy4flash
(Tree-8 SpecInfer) are separate design points.

Survivors (the do-first queue, §6): the greedy-card guard (a **proven defect on our code**,
§4), the cc12.0 env-arm cluster, the conversation-cache arms measured against our own log,
PEER_STALL_S, and the offline F-228 arithmetic. Everything else was falsified or dropped with
mechanisms — see §4 for the drop shelf and §5 for what the adjudication itself taught.

## 4. What has NOT been achieved

**The +10-25% rows class is priced, armed, and not cashed.** Dossier §6 items 1-3 (lookahead
stats falsifier; depth-2 quiet re-test ladder; swaps=80 + spec-min-p 0.7) are behind the card with
runbooks written; F-228's per-depth data (depths 3/4/5 offered 35/17/8%, accepted ~85/68/75%)
vindicates the acceptance premise, but the offline SPEC_COST arithmetic and the ladders have not
run. This is the credible path to solo 250-270 and it is entirely unexecuted this week.

**The conc lane past ~312 is unbuilt.** C2 admission overlap (priced "large") and the mixed-form
cache (tiered requant, coverage/byte +29% on paper) are design-stage. The conc wall is the same
host-stage equation at 4.68 rows; no structure lever for it has been pulled since the cohorts win.

**The drafter's ceiling is unowned.** The re-harvest + retrain cycle (dossier item 8, the
multiplier on every rows-class win) has not run; `harvest/records/windows.npy` is absent and is a
hard prerequisite for the trainer (W=512/K=6/seam-law per SVG-CUSTOM:10477). Until it is rebuilt
over the 11 shards, the drafter-quality leg of the composition is blocked.

**The TTFT lane is measured-adjacent but unswept.** The external rig reads our prefill flat
(1,940 → 1,922 tok/s, 4K → 32K) against their 1,852 → 2,917 — directional only, unverifiable, no
artifacts. Our STRATA_PREFILL_TIMING instrument exists (prefill.cpp:4995-5001) and the 4K/16K/32K
sweep does not. B2 (long-prompt prefill scaling) and B3 (routing-aware cold prefetch) from
standing-line Tier B remain open.

**The formal quality suite is specified, not built.** F-450 names it (5-task coding class +
short factual, greedy + fixed-seed, byte-identical greedy); no suite file runs it yet. Until it
exists, quality adoptions rest on the MMLU interleaves and the judge batteries — thinner than the
rule the operator set.

**The persona standing is not fully proven.** The commit record says it: MMLU equivalence at 0.3
is "statistically indistinguishable" at n=70 (p=1.000 rules out a 0.45-sized effect, not a
≤2-question one); the ABAB interleave vs the pre-adoption standing was NOT run (single boot,
same-hour control only); a second 0.3 gate is owed. OFF 67/70 is the best standing reading, but
the standing's quality proof is one thin gate strong.

**Ten commits are unpushed** (origin/main = 825ee1183, HEAD = 66a0c6beb): the two adjudications,
the E DROP revision, the entire persona arc, and the store closure. Law 14: un-pushed work does
not exist. This is the largest hygiene debt on the board.

**The drop shelf (banked negatives, each with its mechanism and re-open clause)** — dossier
§§10-11, all committed (2991098cb, 3647af45f):

| Item | Verdict | Mechanism |
|------|---------|-----------|
| POOL_DEP (pool dependency overlap) | compose-only | banked f325 data: serial quant 0.009-0.015 ms/round → win 0.4-0.7% of window, below the +4% bar standalone |
| Parallel doorbell quant | DROP | decode exposure 0.036-0.090 ms/window banked; prefill ceiling ~0.5% |
| CPU-share link_rule + NT | DORMANT | share gate requires `o.batch <= 0` (generate.cpp:17586-90); at parallel:6 the CPU share is OFF entirely |
| VBMI byte-lookup dots | DROP | our dot is byte-bound: F-375 45.7 GB/s = 93-96% of DRAM ceiling, above their pending "from-memory" 40.5; single-token-vs-1..8-token M1 hazard |
| L2_PF | DROP | measured sm_120 negative (−8-12% on 1×5090 IQ3_S, #6431) |
| ROUTE_PRIOR | DROP | owner-reverted upstream (#4402) |
| ADAPT_FETCH=2 as default | DROP | upstream rule applies ≤16 GiB only; not bit-exact (KL 0.948 [0.830, 1.063]) |
| Shared-prefix tokenization | DROP | already ours (PromptEncoder, f13a870d5) — my claim was wrong, §5.3 |
| RAM-diet family | DROP | ~0 on our geometry |
| R1-R6 damaged-opener | SHELVED | until our logs show the class |
| d35353903 MTP-greedy flip | DROP | blast radius |
| VERIFY_DEVICE_HITS | DROP | F-222/T1.1 owns the class (STRATA_PLAN_PARTIAL); verbatim port inert at explicit pcie_frac 0.35 |
| KVA (KV asymmetric) | Tier C quarantine | KL 0.018-0.020 nats; ladder written, operator authorization required |

## 5. What has been learnt

### 5.1 The falsifier-first rule cashed at scale

The adjudications' decisive moves were all free: the greedy defect cost one REPL session; the E
(VBMI) idea died against F-375's existing binaries; POOL_DEP's true price came from the banked
f325 stair files, not a new boot; D's dormancy came from one read of the gate condition. Nine
claims resolved for zero boots. Before this window, the method's cheapest-falsifier law was a
discipline; after it, it is the demonstrated difference between a two-boot queue and a two-week
queue.

### 5.2 External intel is re-pricing, and their numbers are theirs

PolyStrata's headline class (BW-C trio, decode +2.0%) does not transfer wholesale: the L2_PF leg
of their trio is a measured sm_120 negative on our exact card+pack. Their 40.5 GB/s quant-dot
figure is their own "from memory" note, pending on their side, against our measured 45.7 GB/s
byte-bound. Their env defaults exist because their default card is cc12.0-era-adjacent; ours is
sm_120 and our tree has no sm_120 nv_defaults table at all (nv_defaults.cpp = sm_86/sm_60) — which
is precisely why the composed-verified set is worth ~15 LOC here. The lesson generalizes: a
production precedent on your component class outranks a paper number on someone else's, and the
transfer check is a measurement, not an argument.

### 5.3 Three self-retractions, on the record

1. My "shared-prefix tokenization LACK" — refuted; the commit I cited for porting is in our own
   history. Cause: grep with wrong symbols, against a tree another session had just extended.
   Law 18 (one capability, one owner) now includes *inventory before claiming absence*.
2. My POOL_DEP "~+10%" pricing — falsified by banked data (0.4-0.7%). Cause: carried forward from
   a compaction's 1.35 ms figure instead of re-reading the f325 files.
3. My ADAPT_FETCH pricing — revised from "=2 bit-exact" to "=1 with a KL canary" after reading the
   upstream guard (≤16 GiB) and the measured KL.

### 5.4 The quality axis has its own physics

The persona arc earned laws the perf axis never needed: dose lives in the block's read domain
(harmonic, not arithmetic — a mechanism, not a knob); a register effect has a where/when structure
(phase-scoping beat dose-lowering); composition is a first-class question (one cvec table per boot
made persona-vs-refusal a false choice until two tables existed); and a quality change needs its
own interleave discipline — per-request ON/OFF in one boot (McNemar), not an MMLU run per boot.
The determinism canary (`shadow_misses`) is the bridge between the axes: without it, quality
comparisons across quantization mixes are noise wearing a p-value.

### 5.5 The record defends itself

Twice this window the banked record overturned a live claim of mine (F-375 vs the VBMI premise;
f325 vs the POOL_DEP pricing) and once it overturned an external's (L2_PF). Banked negatives with
mechanisms are the cheapest staff in the campaign. Corollary learned the hard way in the store
closure: the same record also catches *structural* debt — the trainer had been resolving zero
record shards and nobody had noticed, because nothing executable checked the closure.

## 6. What is not clear

1. **A2 vs T63 (owed operator ruling).** standing-line A2 says route tail skip stays 0 everywhere
   (external read: off was faster AND higher quality; upstream ships 0); the standing recipe
   carries ROUTE_TAIL_SKIP=7 from T63, measured golden-identical with MMLU 98.6/95.7; upstream
   flipped the default off in 0.1.63. Both cannot be the standing rule. Unresolved on the board.
2. **Whether the persona standing survives the formal quality suite.** n=70 MMLU cannot prove
   equivalence; the F-450 suite does not exist yet; the second 0.3 gate and the ABAB-vs-standing
   interleave are owed.
3. **The drafter's acceptance ceiling.** M3b era-gap (acceptance drift across eras) is unexplained
   (dossier §3, item 7); the depth acceptance curve is measured at the offer side (35/17/8%,
   accept ~85/68/75%) but the SPEC_COST arithmetic that would price depths 3/4/5 is offline and
   not run.
4. **The coverage-axis correction.** Dossier §2 re-prices F-418's fit (the coverage axis, not the
   residency axis, may own the cache-hit curve). One boot names it (16k-slot falsifier, expected
   REFUSE). Not run.
5. **The peer-stall leak.** A stalled slot is suspected of silently costing 1/6 concurrency
   (the PEER_STALL_S port, ~25 LOC, would make it observable). Existence and rate on our box:
   unmeasured.
6. **The external decode-gap regime.** The 3.21× decode gap that motivates standing-line Tier A
   comes from a 16 GiB laptop at 3.08 GiB residency; we hold 94.9-97.6% VRAM share on 32 GiB.
   Whether *our* shadow_misses steady-state is already ~0 (making A1 moot here) is exactly what
   the falsifier checks; not run.
7. **First-request cost.** warm_start's pin-core port is gated on a first-request-vs-mid-session
   DONE-line falsifier that has not run; the residual win may not read.
8. **Ownership of the in-flight tree.** The working tree carries a live estate-consolidation
   session's modifications (ROADMAP, specs, battery-lib, vector-lab, launchers, configs). Its
   journal + commit at round close (law 14) is that session's obligation; this paper takes no
   position on its content.

## 7. The queue (card-gated; the engine serves on 8081 — no boots this window)

Order is falsifier-cost first (laws 4/15); MMLU gate binds only token-stream-affecting changes;
server-only changes need no binary ladder (F-19 precedent).

1. **[0 boots] F-228 offline SPEC_COST arithmetic** — depths 3/4/5 offered 35/17/8%, accepted
   ~85/68/75% vs step_ms = 8.4 + 1.57T. Decides whether the depth ladder is worth its five boots.
2. **[0 engine LOC] Arm tool_call_recovery:true** (frontend.py:931-1469, wired :4346, config
   :9141) + run their rcall_specimens.json as the parser falsifier (~60 LOC adapter, no GPU).
3. **[1 boot] Greedy-card guard port** (~70 LOC from d7ce708ba only; NOT d35353903; env
   kill-switch). PROVEN defect: REPL on our code — temperature-0 emits `top_p=0.8 top_k=20
   penalty_present=1.5 penalty_last_n=64`; sampler.cu:1259-1269 argmaxes over penalized logits;
   F-400's prewarm replay posts temp 0.0 through the same path. Pair with arming
   reasoning_loop_recovery (presence 1.5 is our incidental anti-degeneration medicine; F-370/F-371
   hazard). MMLU + M3b mandatory; output jump pre-registered.
4. **[1 boot each] cc12.0 env arms** — EXP_MODE=8, NO_MMVQ_ROWS2=1 (bit-exact), HC_UPMIX=1
   (+CHECK=64 built-in), LFUSE=1, PLE_BATCH=1 (12.8 hazard N/A on our 13.4), QFUSE=1 (watch
   partial-VRAM decode, #6583). Compose survivors via `campaign/binary-ladder.sh`; land the
   composed set as the ~15-LOC sm_120 nv_defaults table + test. Q80_ROWS1 scoped (10 LOC, mirror
   e6efa2b16) only if NO_MMVQ_ROWS2 reads.
5. **[1 diagnostic boot] STRATA_PREFILL_TIMING=1 at 4K/16K/32K** (+5-15 LOC floor print) +
   PEER_STALL_S port (~25 LOC). Names the TTFT lane and the stall leak.
6. **Conversation-cache ladder** — prompt-cache 12 → park_gain 256 → rewind_keep; gate: the 193
   full re-reads / 4,820 events count must fall. Watch-only for quality.
7. **warm_start falsifier** (first-request DONE line vs mid-session) → pin-core port (~120-160
   LOC) only if the residual reads.
8. **Compose-class** (only inside a passing composition): POOL_DEP (~70-85 LOC, 0.4-0.7%),
   DEQUANT_SIDE (~80-120 LOC + parity).

After the arms: the F-450 quality suite gets built (it is a prerequisite for every future
adoption verdict), the second 0.3 MMLU gate runs, and the depth ladder (§6 item 1's verdict)
decides whether the +10-25% class is cashed or re-priced.

## 8. Provenance

- Canonical dossier: `campaign/research/levers-2026-11-20-evening.md` §§0-11 (cache study §7,
  PolyStrata delta §8, upstream delta §9, serve adjudication §10, engine adjudication §11 + E
  addendum). Commits 49fcf265e, 825ee1183, 9131eb315, 2991098cb, 3647af45f.
- Quality arc: campaign/research/2026-11-23-vector-fresh-eyes-review.md,
  2026-11-26-vector-v2-research-and-fix.md, campaign/vector-lab.md, campaign/results/vec-2026-11-23/;
  commits 0bd36187e, ec999a3bc, 3c524a5dd, 7e6c2878a, 7abdc0a0f, ba13d1acb.
- Standing rule: campaign/standing-line.md (F-450). Tracker: campaign/tracker/lanes.csv +
  build-trend.py. Architecture: SVG-CUSTOM.md F-432..F-450. Store closure: 66a0c6beb.
- All performance numbers: ours from probe-lane batteries on this box (means, warm legs 2+,
  interleave-verified) unless prefixed "external/theirs". All external numbers quoted with their
  sources in §§8-11 of the dossier and used as direction only.
