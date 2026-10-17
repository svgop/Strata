# T14C — FastMTP-class drafter retrain: acceptance arithmetic and integration gates

Date: 2026-10-17 (researched 2026-10-16/17). Scope: the acceptance→t/s model for the T7.B.1
FastMTP-style drafter retrain on this engine, the integration-gate ladder for adoption, and the
literature calibration (FastMTP / EAGLE-3 / DeepSeek-V3 MTP). Read-only research; nothing was
built, booted, or modified.

---

## Summary

1. **The engine's own banked data calibrates a complete window model.** From the s35 3v3
   (min-p 0.35 arm vs 0.5 ctl, campaign/results/s35arm-*.txt vs s35ctl-*.txt, row-[15] accept
   accounting pooled over 3 interleaved rounds): the marginal offered verify row costs
   **c_row(M1) ≈ 3.56 ms** (range 2.5–4.6 across calibration methods), and carried **0.604
   expected tokens** — net −1.75% M1, reproducing F-228's verdict from first principles. The
   6-conc falsifier (f6m0) independently gives c_row(conc) ≈ 4.62 ms/row. One law: **a marginal
   verify row costs ~3.5–4.6 ms in every lane** (the T-scaled graph of F-228), not the 0.55 ms
   T13A had assumed.
2. **Crossing rule:** a depth extension pays only while **ΔC/Δoffered > c_row·C₀/W₀ ≈ 0.72
   expected tokens per extra offered row** (M1, standing calibration). The 0.35 arm's marginal
   rows carried 0.604 → negative, exactly as measured. A retrained drafter flips the 0.35
   re-sweep positive at **+5pp of deep-depth marginal acceptance (breakeven), ~+10–15pp for a
   clear win** — or for free if the kernel tranche (hc norm-fold etc.) takes c_row to ≤2.8.
3. **Scenario table (central estimates):** (a) per-proposal 78→85% at fixed shape: **+5.7% M1,
   +2.8% pair, +2.7% conc** (windows flat — acceptance-side gains are pure); (b) offered
   2.2→3.5 at today's marginals: **+2.0% M1 central, range −5%..+10% over c_row uncertainty**
   — breakeven-at-best, confirming F-228; (c) both: **+8.7% M1 central (−3%..+17%)**; batch
   lanes get only the (a) component at the current 2-row batch shape; (d) FastMTP mapped:
   their *shape* (K=3 unconditional) is **−19% on our engine** (our rows are 20–30× more
   expensive relative to window than their A10 dense-7B rows); their *recipe* (acceptance lift
   at fixed shape) maps to scenario (a). T11.R3's "FastMTP-scale → ~237" held the window flat
   while adding tokens; re-priced, the retrain alone lands **~223–229 M1**, and the corrected
   **250 path = kernel tranche (−1.6 ms) AND retrain (mean marginal ≈ 0.86–0.87)** — neither
   alone reaches it. On the current window, 250 is arithmetically impossible at any offered
   depth (C_needed exceeds the m=0.95 ceiling by 3–6%).
4. **Gate ladder:** G0 greedy bit-exact mimic (binary, minutes) → G0 M3d stair read (v1
   gates: pos-2/3 +≥8pp, pos-0 ≤1pp regression) → G1 one config-boot battery (+4% M1
   pre-screen) → G2 ABAB 3v3 (adoption bar, +4%) with batch telemetry → G3 MMLU-Pro 2× pooled
   + mimic band → G4 F-229 ungated re-boot (crosses at offered ≈ 4–5) → G5 min-p re-sweep
   (crossing rule above). **Most likely to fail: G2's +4% M1 bar** — the defensible central
   gain is +5.7–6.4%, and the confidence-coupled chain extension taxes 1–2pp per +0.3 offered
   rows unless deep marginals hold ≥0.72/row.

---

## 1. The acceptance → t/s model

### 1.1 Definitions (from code, src/program/generate.cpp)

- A serial verify window carries T rows = 1 seed + (T−1) offered drafts;
  `draft_offered += T−1`, `draft_accepted += a` (a = accepted drafts per window;
  generate.cpp:13017–13018). **Committed per window C = 1 + accepted/window.**
- Per-request "accept %" = accepted/offered over draft rows only (unconditional).
- M3d per-depth counters (STRATA_SPEC_DEPTH=1, generate.cpp:13006):
  `dp_tested[d]` = windows whose chain reached depth d+1 unrejected; `dp_ok[d]` = depth-(d+1)
  drafts accepted. So the printed marginals are **conditional** marginals
  m_k = P(accept at k | offered at k, chain unbroken through k−1). Drafts past the first
  rejection are verified but not counted in dp_tested.
- Batch lane (generate.cpp:10019–10419, F-231): per-slot row widths by draft staleness
  ("stale-draft slots taking one row"); rescue counts as accepted in telemetry
  (generate.cpp:10264).

### 1.2 Banked baselines (ground truth)

| Lane | committed/window C | offered/window | window W | t/s |
|---|---|---|---|---|
| M1 standing (standing-final-f224 class) | 2.47 | ~1.9–2.2 | 11.4–12.0 ms | 200–216 (mid 211) |
| M1 stair boot (STRATA_SPEC_DEPTH=1, stair50.log M1 probe req) | 2.81 | 2.39 | 12.9 ms | 218.5 |
| s35 3v3 ctl (pooled r1–3, row-15 accounting) | 2.77 | 2.35 | 13.7 ms (implied) | 201.5 |
| s35 3v3 arm (min-p 0.35) | 3.06 | 2.83 | 15.5 ms (implied) | 198.0 (−1.75%) |
| pair | 2.88 | 3.15 rows | 14.2 ms | 203 |
| 6-conc | 5.69 | 6.55 rows | 22.32 ms | 255 |

F-228 offer distribution at min-p 0.5 (task ground truth): s = P(d≥k) = [1.00, 0.63, 0.35,
0.17, 0.08, ~0], offered ≈ 2.20; conditional marginals m = [0.77 (per-proposal baseline),
0.86, 0.85, 0.68, 0.75] for depths 1–5. Raw stair50 cumulative (343/374 240/269 164/194
117/134 71/89) gives m = [0.92, 0.89, 0.85, 0.87, 0.80] on its boot; stair35 gives
[0.92, 0.87, 0.80, 0.79, 0.71] — the task's 86/85/68/75 sits inside this boot spread. Note
the two baselines reconcile: the stair boot runs hotter traffic (temp-0.7 canary) and its
30% zero-draft windows + typical-rescue fold the 2.81 boot figure down to the standing 2.47.

### 1.3 The two laws of window cost

**Tokens:** C(s, m) = 1 + Σ_k s_k·m_k. Baseline model: C₀ = 1 + 0.77 + 0.63·0.86 + 0.35·0.85
+ 0.17·0.68 + 0.08·0.75 = **2.785** at offered 2.23 (matches the 2.77–2.88 measured class;
the standing 2.47 is the same engine on colder traffic — deltas, not absolutes, are taken
from the model).

**Cost:** W = W₀ + c_row·(offered − offered₀). Calibrations of c_row:
- s35 3v3 direct decomposition (best): ΔW/Δoffered = (15.46−13.74 ms)/(2.833−2.351) =
  **3.56 ms/row**. Cross-checked: net arithmetic closes (ΔC +10.5%, ΔW +12.4% → −1.7%,
  measured −1.75%).
- 6-conc falsifier (f6m0): (22.9−19.71)/(6.62−5.93) = **4.62 ms/row**.
- Probe "ms/window" meter gives 1.67 ms/row but that meter disagrees ~10% with headline
  walls — discarded as a metering artifact (noted, not used).
- Range carried: **c_row = 2.5–4.6, central 3.56** (M1), 4.62 (conc). Includes the
  draft-step tail (~0.05 ms GPU/step, F-229) and pool second-order growth.

**Elasticities (windows flat):** M1: +0.80%/pp of per-proposal acceptance; 6-conc: +0.44%/pp
(stale-slot haircut k=0.63: a_eff = 0.63·p); pair: +0.46%/pp.

### 1.4 Crossing rules (the answer to "where does it flip")

Net: Δt/s = ΔC/C₀ − c_row·Δoffered/W₀. Depth extension pays iff
**marginal tokens/extra row > c_row·C₀/W₀ = 3.56 × 2.77/13.74 ≈ 0.72** (M1, standing
calibration; conc: 4.62 × 5.69/22.32 ≈ 1.18 tokens/row — batch depth is far more expensive
per row). Equivalently: one accepted token is worth W₀/C₀ ≈ 4.96 ms; a marginal row must
return ≥ 0.72 of them to cover 3.56 ms.

- 0.35 arm measured blend: 0.604 tokens/row → negative ✓ (reproduces F-228).
- Breakeven deep-marginal lift at the 0.35 anchor (deep offered mass ≈ 1.04 rows/window):
  **+5pp**; +10pp → +1.6% M1; +15pp → +3.3%; +20pp → +4.9%.
- Kernel tranche alone (−0.8 ms window) flips the standing 0.35 arm to ≈ +2–4% **before any
  retrain lift** — the gate-tune question and the kernel question are the same question.

---

## 2. Scenario table

M1 deltas vs 211 t/s mid (window 11.7 ms, C 2.47 standing / 2.785 model); pair vs 203;
conc vs 255. "(a) fixed shape" windows are flat, so Δt/s = ΔC/C exactly.

| # | Scenario | M1 t/s (Δ) | pair | 6-conc | Arithmetic |
|---|---|---|---|---|---|
| 0 | Baseline | 211 (200–216) | 203 | 255 | C=2.47@11.7ms; 2.88@14.2; 5.69@22.32 |
| a | per-proposal 78→85% (×1.09, fixed shape) | **223 (+5.7%)**; flat+8pp variant +6.4% | **209 (+2.8%)** | **262 (+2.7%)** | m'=[.84,.94,.93,.74,.82] → C 2.785→2.945 (+5.8%); batch rows fixed (staleness-set), a_eff=0.63p: p .666→.726 → C 5.69→5.83 |
| b | offered 2.2→3.5, same marginals | **215 (+2.0%)**, range −5%..+10% over c_row 2.5–4.6 | n/a¹ | n/a¹ | s'=[1,.85,.70,.50,.30,.15] → ΔC=+0.99 (+35.5%); ΔW=3.56×1.30=+4.6ms → 18.3ms; at c_row 2.5: +10.1%; at 4.62: −5.0% |
| c | (a)+(b) | **229 (+8.7%)**, range −3%..+17% | (+a only) 209 | (+a only) 262 | C 4.022 (+44.4%), W 18.3ms; deep marginals at the (a)-lifted profile |
| d | FastMTP mapped | see §2.1 | — | — | three mappings below |
| — | Acceptance-only ceiling (m→0.92 / p→0.90, fixed shape) | 222 (+9.6%) | 225 (+10.7%) | 281 (+10.4%) | C=3.05; batch C=4×(1+0.63·0.90)=6.27 |

¹ The batch window offers ~1 draft row/slot (2-row shape at 66.6% slot acceptance; rows are
staleness-set per F-231). Deeper batch chains need the F-231 nrows[] row-width extension
first; until then (b)/(c) depth gains are M1-lane only.

### 2.1 Scenario (d): FastMTP numbers mapped to our scale

Paper facts (arXiv 2509.18362, Table 1 + Fig. 3, MiMo-7B-RL dense 7B, single 210.8M-param
MTP head, A10): vanilla MTP K=3 τ=1.83 (per-depth acceptance ≈70/10/~0); FastMTP self-distill
τ=2.73, acceptance **80% @ k1, 56% @ k2, 36% @ k3**; +FR τ=2.66 at 2.03× vs no-spec
(31.55→64.12 t/s), **+49% tokens at fixed window vs vanilla MTP** (38.04→57.01 t/s, window
~48 ms flat); K=3 optimal, "beyond K=3 marginal acceptance doesn't cover drafting overhead."

Three mappings:
1. **Shape map (their offered shape on our engine):** K=3 unconditional, m=[.80,.56,.36] →
   C=2.72 at offered 3.0, W = 13.74+3.56×0.77 = 16.5 ms → **165 t/s (−19%)**. Their rows cost
   ~17% of window (5.4/32 ms on A10 dense-7B); ours cost ~26% (3.56/13.7). Do not ship their
   shape; ship their training.
2. **Recipe map (their acceptance lift on our gated shape):** their lift is depth-skewed
   (k1 +10pp, k2 +46pp, k3 +36pp); applied to our profile with caps ≈ scenario (a) to
   (a)+: **+6–9% M1**. This is the defensible central expectation for the retrain.
3. **Composite (their recipe + depth re-unlock at offered 3.5):** deep marginals →~0.90,
   C=4.20, W=18.3 → **230 (+13.4%)** — the optimistic edge, only if BOTH the deep conditional
   products lift AND the extension respects the 0.72 rule.

**T11.R3 re-pricing (material correction).** T11.R3's decision math ("2.78 banked/13.2 ms ≈
205-210; RH-scale → ~227; FastMTP-scale → ~237; depth re-unlock → 247-255; 250 needs
AL ≈ 3.25-3.4") held W flat while adding up to +0.6 tokens. With c_row = 3.56: AL 3.3 arrives
at offered ~3.0 (W ≈ 16.5) → 200-228, not 247-255. Corrected composition: **250 t/s requires
C/W = 0.25; on the current window this is impossible at ANY offered depth** (C_needed exceeds
the m=0.95 ceiling by 3-6% at offered 2.2/2.8/3.5). With the T13A micro-tranche (−1.63 ms
midpoints): offered 2.23 needs mean marginal **0.861**; offered 2.8 needs **0.867**. The 250
path = kernel tranche AND retrain, each roughly half of the distance. The retrain alone has a
hard ceiling of **~222-229 M1** (acceptance-only ceiling at fixed shape), and ~235 even with
m=1.0 at today's rows.

---

## 3. Integration gates (adoption ladder for the retrained drafter)

Every arm is a config-json (new --mtp path) through battery-config.sh, which sources the
launcher's standing env (F-214 lesson) — a retrained GGUF is otherwise just another arm.
Order is by cost; each gate has a cheap falsification.

**G0.1 — Greedy bit-exact mimic (binary kill, ~1 boot + 1 min).** bit-check.py (3 prompts,
temp 0) retrain-boot vs standing-boot. Outputs are verified-exact (F-230 law: wrong drafts
are REJECTED, not emitted), so greedy text MUST be byte-identical to ctl. Any divergence =
draft leaking into the commit path (engine bug class, not a drafter property). Reject
immediately, do not pass G0.2.

**G0.2 — M3d stair read (the drafter's own v1 gate, 1 boot).** STRATA_SPEC_DEPTH=1 boot, M1
probe; read dp_ok/dp_tested marginals + offered/window + accept accounting. Gates (T11.R3
v1, restated in model units): depth-2 and depth-3 conditional marginals **+≥8pp each**;
depth-1 **≤1pp regression** (a retrain that trades k1 for k2 is a net at fixed shape —
elasticity 0.80%/pp — so k1 is sacred); offered/window reported so G5's crossing rule can be
pre-computed. Falsification: if pos-2/3 lift < 8pp, the arm dies here for the cost of one
boot (no 3v3 spent).

**G1 — One config-boot battery (speed pre-screen).** probe-lane.py full lane; require M1
≥ +4% vs the same-day ctl anchors before spending a 3v3 (the standing adoption bar; F-226
adopted at +4.06%, F-231 held below-bar at +1.81%). Read M3b JOINTLY with offered/window:
a retrained drafter with a deeper chain will mechanically compress the acceptance RATE
(s35: 75.2→72.7% while C rose +10.5%) — M3b is a drift gauge here, not a quality gate.
True flags: M3b < ~70 (below the s35-arm floor) or depth-1 marginal regression.

**G2 — ABAB 3v3 (adoption instrument).** 3 interleaved arm/ctl boots (abab-run.sh pattern,
binary swap only inside the script). Bar: M1 ≥ +4% with arm ahead in all/most rounds, pair
and conc no-regression. Carry the batch telemetry lines ("MTP proposals accepted X of Y",
slot acceptance baseline 66.6%) — the pair lane carries F-231's unresolved −3.61% flag and a
188-209 band; a retrain that only lifts M1 while pair sits at 192 twice is the F-231 pattern
and must not adopt. Falsification: ctl-round outliers (the r2 class) are watched, not chased.

**G3 — Quality gates (different-bits law).** MMLU-Pro 2× (mmlu-pro-gate.py, 70q): pooled
arm ≥ pooled ctl − 1σ (σ ≈ 5.2 pts/run, ≈ 3.7 pooled). Run arm and ctl fresh in the same
window (the matched-window form the gate script exists for). Runner detail from F-230: needs
.venv python (system python3 lacks `datasets`) — fixed before the next gate. The mimic lane:
mimicbench 64 in the 95.3-98.4 band on the arm boot. Rationale: at temp 0 the retrain cannot
change outputs (G0.1 proves it); G3 guards the temp-0.1 coupled-sampling interaction and
accidental engine regressions — expect PASS, and a failure means something is wrong beyond
the drafter.

**G4 — F-229 ungated re-test (free, one boot).** Condition: offered/window ≥ ~4 after the
retrain. Model: ungated net = host-bubble saving (~0.1 ms, F-229's own decomposition) −
waste 0.05 ms ×(6−O) − burst residual; crossover O* = 4.0 under the fixed-residual read,
O* ≈ 5.2 if the residual scales with discarded launches (both under-identified from one 3v3;
F-229's note says "chain ~5 of 6"). Expect ≥ +1.5-2% M1 only at offered ≳ 5. One
STRATA_DRAFT_UNGATED=1 boot on the retrain answers it; stays default-off unless clearly
positive.

**G5 — min-p re-sweep (the 0.35 flip).** One STRATA_SPEC_DEPTH=1 boot at 0.35 (and 0.42)
on the retrained drafter. The crossing rule, measured live from the accept accounting:
**ΔC/Δoffered > c_row·C₀/W₀ = 0.72 tokens/row** (today's drafter: 0.604 → negative; deep
marginals +5pp = breakeven, +10-15pp = the +1.6-3.3% zone). If the kernel tranche
(hc norm-fold, bf16-native) lands first (c_row → 2.8), the threshold drops to ~0.56 and the
0.35 gate-tune flips without any retrain. Confirm with one 3v3 only after the stair read
clears the rule; never re-arm 0.35 on the old drafter (two measured negatives bracket 0.42,
F-228).

Adoption = G0.1 pass ∧ G2 bar ∧ G3 pass, with G4/G5 as follow-on boots on the adopted base.

---

## 4. Literature calibration

| Source | Drafter | Scale | Depth profile / acceptance | τ or tokens/step | Speedup | Same-family single-layer like ours? |
|---|---|---|---|---|---|---|
| **FastMTP** (arXiv 2509.18362, Tencent, Sep 2025; OpenReview J7xDwZSyI4) | single MTP head, position-shared weights, DeepSeek-V3-style, trained with recursive teacher-forced CE (α_k = β^{k-1}/Σ, β=0.6, K=3) on 389.4K self-distilled samples; <1 day on one H20 | MiMo-7B-RL (dense 7B, 36 layers); head 210.8M (<3%) | vanilla: ~70/10/~0 (k1/k2/k3); FastMTP: **80/56/36** | τ 1.83 → **2.73** (K=3, A10) | 1.81-2.03× vs no-spec; **+82% over vanilla MTP**; +49% tokens at fixed window | **Yes** — closest analogue |
| **EAGLE-3** (arXiv 2503.01840, Mar 2025) | single fresh decoder layer consuming 3-layer fused target features; ~8× EAGLE's data (ShareGPT+UltraChat-200K, target-generated) | Vicuna 13B, LLaMA-3.1-8B, LLaMA-3.3-70B, DS-R1-Distill-8B | tree drafting (~60 nodes, depth ~6), not a chain | τ 5.8-6.6 @ T=0; 4.9-5.7 @ T=1 | 3.0-6.5× (up to 6.5× Vicuna/HumanEval); +20-40% over EAGLE-2; ~1.4× EAGLE-2 class | **No** — separate architecture, feature-fusion input, tree verifier; τ not comparable to chain-6 |
| **DeepSeek-V3 MTP** (arXiv 2412.19437, Dec 2024) | 1 MTP module (same-family block, shared embed+head), depth-1 at inference | 671B MoE (37B active) | **85-90% second-token acceptance** | ~1+0.85-0.90 tokens/step | **up to 1.8× TPS** | **Yes** (architecture), but serving at datacenter scale where verify rows are relatively cheap |
| Production trims | EAGLE-3 in vLLM | — | max chain 2 in the vLLM throughput test; ~2.4 accepted/round in practice | — | converges to bs≈1 gains (1.75× at bs=2 → 1.01× at bs=56) | — |

Classic formula anchor (Leviathan et al., arXiv 2211.17192): E[tokens] = (1−α^{γ+1})/(1−α);
speedup = (1−α^{γ+1})/((1−α)(γc+1)) — our §1.3 is this formula with position-dependent α,
an offer distribution s_k (the min-p gate), and a measured per-row marginal cost c_row
instead of their uniform-window assumption.

Production recipe: Red Hat `speculators` 0.6.0 (developers.redhat.com, Sep 2026;
github.com/vllm-project/speculators; docs: "MTP finetuning starts from the model's
pre-existing MTP layers", Qwen3-Next/Qwen3.5 supported) — the exact head class T11.R3
validated as ours. Example artifact: RedHatAI/Qwen3-8B-speculator.dflash.

**What is realistic for our 5.2GB single-layer same-family drafter on one 5090:**
- Training is firmly in the demonstrated regime: FastMTP trained a 211M-param head on <1
  H20-day with 389K self-distilled samples; RH's recipe gets usable deltas from ~5K samples
  (T11.R3's read); head-only fits a 5090 (8-14 GB, 2-6 h/arm). No cloud needed.
- Expected depth-marginal outcomes, translated to OUR conditional (gated) units: depth-1
  0.80-0.90 (we are at ~0.77-0.92 depending on boot — parity to +5pp); depth-2/3 +8-15pp;
  depths 4-6 (the retrain's actual target, β≈0.8 weighting): from 0.68-0.75 to 0.80-0.92 IF
  the recursive objective lifts conditional products the way the paper's k2/k3 lifts show.
- Token-side expectation on our engine: τ/C gains of +6-16% (scenarios a→c), NOT the paper's
  +49%-at-fixed-shape (their vanilla drafter was far worse than ours: 10% at k2 vs our 86%
  conditional at depth 2 — our gate already harvests what their vanilla lost; the retrain's
  headroom is at the DEEP tail and in making the gate's trims rarer).
- EAGLE-3's τ 5.8-6.6 is not a target for this engine: reaching it means ~10-20 verify rows
  at c_row ≈ 3.5 ms — W would double to reach C ≈ 4-5; net negative. The realistic
  literature-anchored endgame on THIS cost structure is C ≈ 3.0-3.4 at W ≈ 12-14 ms
  (post-tranche), i.e. 230-250 t/s M1.

---

## 5. Contradictions and their resolution

1. **T13A's marginal-row estimate (~0.5-0.6 ms/row) vs F-228's verdict.** Resolved
   empirically: 3.56 ms/row (s35 net decomposition, closes to −1.75% exactly) and 4.62
   (conc falsifier). The T-scaling F-228 describes (hc lines + attention + head re-run per
   row) is the mechanism; hits (0.56 ms/row) are only a minority share of it.
2. **Probe "ms/window" meter vs headline walls** (~10% disagreement; the two give c_row 1.67
   vs 3.56). Resolved in favor of headline-wall calibration (the metric the campaign ranks
   by); the meter disagreement is recorded as a known instrument caveat.
3. **Standing C = 2.47 vs stair-boot C = 2.81-2.88 at the same gate.** Boot/traffic variance
   (temp 0.7 canary vs 0.85 chat; zero-draft windows; rescue share). The model uses the F-228
   distribution for deltas and the standing pair for levels; both are quoted everywhere.
4. **FastMTP's k2/k3 acceptance (56/36) vs our conditional marginals (86/85).** Not a
   contradiction: theirs are unconditional in an always-offer-K=3 scheme; ours are
   conditional on the min-p gate's confidence trim. Comparable quantities are tokens/window
   and t/s, which is what §2 maps.
5. **T11.R3's 247-255 "250 path" vs this model's ceiling.** T11.R3 held the window flat while
   adding tokens; with c_row = 3.56 the depth component is heavily taxed. Corrected: 250 =
   tranche AND retrain (§2.1). This amends, not overturns, the GO verdict — it re-prices the
   depth component and moves the expected landing from 237 to 223-229 for the retrain alone.
6. **F-229's crossover:** the journal says "chain ~5 of 6"; the fixed-residual read gives 4.0.
   Under-identified from one 3v3 — G4 boots it once offered ≥ 4 and measures rather than
   assumes.

## 6. Open questions

- c_row(M1) has no single-boot direct measurement; a STRATA_DECODE_TIMING stair at depth 1
  vs 3 vs 5 on the same boot would pin the per-row cost curve (shape: linear in T or
  superlinear?) and shrink the scenario-(b) range from ±7pp to ~±2pp. One boot.
- The 6-conc stale-slot haircut k=0.63 is inferred from telemetry, not measured; F-231's
  nrows[] infrastructure could expose per-slot row widths directly.
- Whether the retrained drafter's offered-chain extension (confidence up → deeper gate
  pass-through) preserves the 0.72 rule is THE empirical question; G0.2's stair read answers
  it before any 3v3 is spent.
- Pair-lane F-231 flag (−3.61%, twice at 192) unresolved — the retrain 3v3 must not adopt on
  an M1-only win while pair sits at the band floor.

## 7. Implications

- The retrain's value proposition on this engine is **acceptance at fixed shape
  (+5.7% M1 / +2.8% pair / +2.7% conc central)**, with depth as a *conditional* second
  chapter gated by the 0.72-tokens/row rule. Selling it as the depth lever (the T11.R3
  framing) overstates it by ~2× on this cost structure.
- **250 t/s needs both the kernel tranche and the retrain**; neither reaches it alone, and on
  the current window it is unreachable at any acceptance level.
- The cheapest de-risking sequence spends boots in this order: G0.1 bit-check (minutes,
  binary), G0.2 stair read (the drafter's own v1 gates), G1 battery, then the 3v3. A drafter
  that fails pos-2/3 +8pp dies for the cost of two boots.
- G4/G5 are the retrain's free compounding options: both cross positive only because a more
  confident drafter changes the constants in their rules (waste shrinks; marginal blend
  rises). Neither should be pre-armed.

---

## Sources

Code and repo (primary):
- src/program/generate.cpp:13006-13019 (draft_offered/draft_accepted accounting; dp_tested/dp_ok
  per-depth increments), :13431-13436 (M3d print), :10019-10419 (batch telemetry, slot rows,
  rescue-counts-as-accepted), :12134 (depth counters decl)
- src/core/mtp.cpp (min-p gate ~:1697; un-gated twin :1631-1665; F-184 coupled sampling)
- src/core/verify.cpp (window graph; T-scaled segments per T13A decomposition)
- campaign/probe-lane.py (M3b canary), campaign/mmlu-pro-gate.py (70q, per-request floor,
  1σ print), campaign/abab-run.sh (3v3 interleave), campaign/battery-config.sh (standing-env
  config boots), campaign/bit-check.py (greedy bit-identity), campaign/depth-lane.py
- campaign/results/stair50.txt, stair35.txt + custom-iq3s-stair50.log/.log (M3d cumulative,
  row-15 request accounting), s35arm-r1..3.txt / s35ctl-r1..3.txt (the c_row decomposition),
  rows-f6m0.txt, abab-dt.log
- SVG-CUSTOM.md F-228, F-229, F-230, F-231 entries (2026-10-17); F-226 adoption bar
  (+4%); M3b 74-79 warm band; mimicbench band 95.3-98.4
- campaign/research/T13A-decode-critical-path.md (window anatomy; the refuted 0.55 ms/row;
  Δms→t/s convention); SVG-CUSTOM-TODO.md T11.R3 (FastMTP GO dossier, decision math re-priced
  here), T11.R2 (byte-exact mimic gate law)

Literature:
- FastMTP: Cai et al., arXiv 2509.18362 (abs + HTML full text; Table 1, Fig. 3) —
  https://arxiv.org/abs/2509.18362 ; https://arxiv.org/html/2509.18362v1 ;
  OpenReview J7xDwZSyI4 ; github.com/Tencent-BAC/FastMTP
- EAGLE-3: Li et al., arXiv 2503.01840 (HTML, Table 1/Table 3) —
  https://arxiv.org/abs/2503.01840 ; https://arxiv.org/html/2503.01840v1 ;
  github.com/SafeAILab/EAGLE
- DeepSeek-V3: arXiv 2412.19437 (HTML) — 85-90% second-token acceptance, up to 1.8× TPS —
  https://arxiv.org/html/2412.19437v1
- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding,"
  arXiv 2211.17192 — E[tokens]=(1−α^{γ+1})/(1−α) — https://arxiv.org/abs/2211.17192
- Red Hat Developers, "Optimize vLLM speculative decoding with FastMTP heads" (Sep 2026) —
  https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads
- vLLM speculators: https://github.com/vllm-project/speculators ;
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp ;
  https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash

Source count: 16 (code 7, campaign data/journal 6, literature/registry 6 with overlaps by
design — papers cross-checked against their HTML tables, not snippets).

## Methods note

Unproductive searches (do not repeat): hunting the nospec log for probe-formatted M1 lines
(the nospec boot predates the probe-lane format; no T=1 window anchor exists in the banked
set — the W(1) intercept is unmeasured, only c_row is); the campaign/results rows-f6m0.txt
carries no batch row accounting (the 5.93/6.62 rows figures live in the F-231 journal entry,
taken as given); battery JSONs are boot configs, not result summaries. Reverse-engineering
the exact pooling behind F-228's stated P(d≥k) (0.63/0.35/0.17/0.08) was abandoned after the
raw stair50 M3d cumulative and the s35 row-accounting proved consistency at the ±0.15-offered
level; the task's stated values are used as ground truth per instructions. The probe
"ms/window" meter was found to disagree ~10% with headline-derived windows and was excluded
from calibration. Web numbers were taken from paper HTML tables (FastMTP Table 1/Fig. 3,
EAGLE-3 Tables 1/3, DeepSeek-V3 §3.3.2-level claims as quoted by the search summary) — the
DeepSeek 85-90% figure was cross-checked across the arXiv HTML and three secondary
write-ups; the "93%" variant in circulation was specifically not found in the report.
