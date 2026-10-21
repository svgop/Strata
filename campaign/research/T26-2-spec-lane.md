# T26-2 — the spec lane, source-level, on both sides: basalt's draft/verify design, the draft_cum refit, DFlash2 multi-tap, DSpark/varlen pricing, and the conditional-graph claim

- Date: 2026-10-21. Scope: source-level walk of the spec/verification lane in basalt (`/home/svg/LLM/data/references/basalt`, read-only), our fork (`src/core/verify.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/verify_kernels.cu`, `src/spec/`, `src/program/generate.cpp`), DFlash2 (patch + config), DeepInfer (scheduler + confidence head), plus the CUDA conditional-node semantics check. CPU-only reading; machine LIVE; no builds, boots, GPU runs, or source edits; only this file written. READ-FIRST respected: T23-{basalt,basalt-portstudy,dflash2,deepinfer,xolotl}.md, T25-2-spec-architecture.md, F-201-mtp/acceptance-theory, F-200-specdec (the cited `F-201-specdec-agent.md` does not exist; `F-200-specdec-agent.md` is the specdec brief — read instead). Line numbers below were read this session unless explicitly attributed to a prior file.
- Standing calibration (given): scoreboard M1 212.5 / pair 202.8 / conc 315.7; acceptance 74.9 %. F-228 (SVG-CUSTOM.md:7852-7864): at min-p 0.5 the chain is NOT capped at ~2 — depths 3/4/5 offered at 35/17/8 % of windows, accepting ~85/68/75 %, mean 2.20 drafts offered, ~2.88 tokens/window; the min-p 0.35 re-stair measured −1.75 % M1 3v3 (198.0 vs 201.5) and M3b −4.8 pp; CLOSED for flat gate-tuning. F-229 (:7866-7878): STRATA_DRAFT_UNGATED −2.73 % M1; decomposition ≈ 3.8 wasted step graphs/window × ~50 µs ≈ 0.19 ms GPU tail bought ~0.1 ms host bubble. F-235: STRATA_SPEC_PROB closed with data (M1 −4.2 %); deep-position acceptance 53.1 % vs 93.6 % shallow.

---

## Summary

All five siblings implement the same three doors (drafter input, offer shaping, acceptance rule) and none touches our contract's verdict rule. This pass verifies, from source, the four mechanisms T25-2 ranked, and corrects three things: (1) the mission's "adapt-every at decay 0.85" is wrong on both constants — basalt's defaults are `adapt_every = 4` rounds and `adapt_decay = 0.7` (serve resident-RAM fast path 0.93), and adapt-every is EXPERT-TIER adaptation, not depth scheduling; basalt's depth scheduling IS the draft_cum keep loop. (2) The conditional-graph band +0.5–1.5 % should be re-priced off our own F-229 decomposition: ceiling ≈ the ungated arm's host-side saving (~0.1 ms/window ≈ +0.9 % M1), realistic +0.3–0.9 %. (3) The keep rule's bin table is argmax-calibrated in basalt and our drafter samples coupled at DT 0.85 — the refit on OUR routing trace is not optional diligence, it is the gate for the whole arm; and our rescue path raises `a` above the exact-match walk, so `observe()` semantics need one decision basalt never faced (§c.4). DFlash2's tap list/formulation is verified exactly from source ([5,15,25,35,45] completed-layer hc-means = 12,800 features); its drafter-internal layout is NOT locally verifiable (safetensors is an LFS pointer, no modeling code in the repo). DeepInfer's scheduler is a cumprod + global top-K over survival probabilities, ref-verified; at our 6 slots and measured flat c_row(T) it has nothing to arbitrate single-stream and sits left of its own measured crossover in conc.

---

## (a) Donor mechanism map — basalt (all file:line read this session at the on-disk pin)

### a.1 The keep/drop rule: `include/basalt/core/draft_cum.hpp` (78 lines, whole file)
- **Bins** (:25-28): `draft_cum_acc(p)` maps a draft's own probability to observed acceptance via 8 bins: p<0.30→0.12, <0.50→0.26, <0.70→0.43, <0.80→0.51, <0.90→0.59, <0.95→0.66, <0.99→0.79, else 0.96. Float32 literals, `__host__ __device__`.
- **Keep test** (:31-36): draft j stays iff `prod_{i≤j} acc(probs[i]) ≥ th`; if `!(th > 0.0f)` it degenerates to the flat gate `probs[j] ≥ min_p`. Products accumulated in float32 in index order.
- **Base** (:39): `kDraftCumBase0 = 0.35f`.
- **Placement scaling** (:44-50): `draft_cum_cpu_mb()` set once at tier finalization (`generate.cpp:4779` = (b_ram+b_top)/n_layers/n_expert/1e6, mean CPU-served expert MB per slot); `draft_cum_scale()` ramps the threshold ×1.0→1.5 as that goes 1.1→3.5 MB. Comment: "Fixed by the pack and its placement, never by timing."
- **Per-request adaptation** (:56-76): `DraftCumAdapt::observe(probs, n, acc)` — expected tokens e = Σ_j prod_{i≤j} acc(p_i) (:69); EMAs obs/exp with decay 0.97 (:66,:70-71); threshold ramps linearly from base toward `0.25·scale` as obs/exp crosses [0.9, 1.1] (:73-74). The window's threshold is fixed before its drafts and shared host/device.

### a.2 Depth scheduling = the keep loop in the window loop
- Single-stream (`generate.cpp:9262-9292`): `T = 1; while (T < S_mtp && draft_keep(dprob.data(), T-1, o.spec_min_p, cum_adapt.th())) ++T;` (:9291) — dprob is the PREVIOUS window's `mtp.draft` output (drafted for this window at :9368). Suffix/lookup windows share the trim at :8402 and the batch lane per-slot at :6717 (`sl.cum` per slot, declared :6384).
- After the window: `cum_adapt.observe(dprob.data(), T-1, a)` + `mtp.set_cum_th(cum_adapt.th())` (:9365-9366; batch :6898/:6957; conc :8648-8649/:8730-8731). `set_cum_th` stores `cum_th_` (`mtp.hpp:79`), read at draft time (`mtp.cpp:1486`: `cum_th_ >= 0 ? cum_th_ : draft_cum_base()`).

### a.3 The draft head and its trajectory (`src/core/mtp.cpp`)
- **Round** (`record_round`, :1082-1113): staged catch-up K/V for the window's T cells (`mtp_stage`, :1090-1097) then `record_forward(T,-1)` (K/V only) + `record_forward(1, ra)` — the full head layer on row `ra = 2·max_t−1` only (:1101-1106); `step_end(0)` (:1110). Row layout: window rows 0..T-1 carry the window's records; the host stages the anchor row's step/pos records (:1088-1096).
- **Chain step j** (`record_step`, :1115-1135): one row at the cell staged in step row `max_t + j − 1`, from the previous step's residual and token; "every full row hands the next its hidden input" (:1128) — the residual is taken by the step-end select, not carried separately.
- **Step end** (`step_end`, :1162-1173): one launch (`mtp_step_end`) writes draft j + its prob to the mapped outputs, the next step's token, and (when a conditional tail is wired) sets the conditional handles; `__threadfence_system()` before the visible token write (`verify_kernels.cu:899-900`).
- **Launch shape** — two shapes, chosen at draft time (:1459-1460 `cond = !cond_off_ && min_p > 0 && max_steps > 1 && capture_cond(...)`):
  1. **Per-step graphs** (fallback and min_p≤0): round graph + one graph per step, host-gated — the ungated twin enqueues all steps (:1503-1535), the gated loop launches step j only while `draft_keep(pr, j-1, min_p, cum_th)` (:1563), spinning on the mapped output between steps (`wait_step`, :1549-1555).
  2. **Device-conditional graph** (`capture_cond`, :1241-1323): ONE top graph = [round as child node] → IF(h1){step-1 body + cond for h2} → IF(h2){step-2 body + cond for h3} → … Handles created per step with default 0, `cudaGraphCondAssignDefault` — "reset to 0 at every launch: a skipped step leaves the next one's handle at 0" (:1255-1258); bodies are captured with `cudaStreamBeginCaptureToGraph` into `np.conditional.phGraph_out[0]` (:1279-1289, `cudaGraphCondTypeIf`, size 1). Launch = ONE `cudaGraphLaunch(cond_exec_[tier_][cp][T])` (:1538-1543); the host still polls each produced draft and breaks on the −1 sentinel the graph leaves when it stops early (:1549-1573, :1573 comment "the graph ended without it").
  - Runtime threshold plumbing: the host writes `h_row_[2] = min_p` bits and `h_row_[3] = cum_th` bits into mapped memory each draft (:1483-1487); the device kernels read them through `min_p_bits[0/1]` — so the per-window adaptive threshold reaches the device by MEMORY, while the `cum_th` kernel parameter (baked at capture as `draft_cum_base()`, :1266/:1294) serves only as the compile-time on/off guard (`if (cum_th > 0.0f)`, :824/:904).
  - Preamble/branching: second-stream branches beside the hidden input (:68, :765, :815), L2 prefetch of the attention projections on a third stream (`pf_fork`, :1139-1153). Failure handling: any capture failure prints and latches `cond_off_ = true` — per-step graphs take over (:1311-1319, :1236).

### a.4 The device-side rule — `src/kernels/cuda/verify_kernels.cu`
- `mtp_draft_cond_kernel` (:819-832) and `mtp_step_end_kernel` (:890-913): `p = probs[*row]`; `keep = p ≥ min_p`; if the rule is on, `c = ∏_{i<j} draft_cum_acc(hist[i]) · draft_cum_acc(p)` over the `hist` buffer (= `m_prob_`, the mapped probs of this window's earlier steps) — "the same products, in the same order, as draft_keep on the host" (:824-828, :904-908) — then `cudaGraphSetConditional(h[0], keep?1:0)` and 0 for the other handles (:830-831, :910-911). All under `CUDART_VERSION >= 12040` (:816, :901).
- `hist` semantics: `mtp_step_end` writes `out_p[j] = p` BEFORE the conditional read (:898), and the next step's kernel reads `hist[0..j)` — i.e., the product includes all of this window's earlier draft probs, matching the host's `pr[0..j-1]` accumulation.

### a.5 Commit of accepted tokens (`src/core/verify.cpp`, `src/program/generate.cpp`)
- Acceptance walk: `a = 0; while (a < T-1 && window[a+1] == outv[a]) ++a;` (:9333-9334) — exact match, position-keyed; the bonus row a is kept.
- `ver.commit(a+1)` (:9345) → `Verifier::commit(n_keep)` (:3043-3083): mapped `h_commit_ = [n_keep, n_keep−1, positions n_keep×i32, commit_epoch]` (:3048-3053), a captured commit graph on its OWN stream (`cm_`), gated by an event recorded after the producing window (:3056-3058); async mode records `commit_done_` and returns without waiting (:3061-3071). Emit = outv[0..a] (:9355-9359). The keep rule changes only T — commit/emit are untouched by it.
- Expert-tier `adapt-every` (context, corrected): default `adapt_every = 4` rounds, `adapt_decay = 0.7` (:495-496); serve resident-RAM fast path decays 0.93 unless `--adapt-decay` given (:5719-5723); ticked on a host thread beside commit+draft (:9343-9344); candidates = missing experts with decayed usage ≥ 2.0, swapped when ≥ victim + 1.5, ≤ `adapt_swaps`, copies async during drafting, `usage *= adapt_decay` after each pass (:9204-9253). **0.85 appears nowhere.**

### a.6 Secondary donors (verified where cited to source this session)
- **DFlash2** (`patches/qwen38-dflash2-capture.patch`): capture at completed zero-based layers, `hidden_states.unflatten(-1,(hc_count,hidden_size)).mean(dim=-2)` appended per tap (:9-16); `set_dflash_layers_to_capture` HARD-ASSERTS `[5, 15, 25, 35, 45]` and prints `features=12800` (:42-49); TP1/CP1 assert (:12). `config.json` `dflash_config`: `target_layer_ids [5,15,25,35,45]`, `num_target_layers 48`, `block_size 8`, `mask_token_id 248077`, `selector_rank 256`/`selector_top_k 16`, two-tap depthwise conv `kernel 2, group 16`, drafter = 5 layers, sliding_window 2048, `is_causal false, use_cache false`, full vocab 248,320. **Not locally verifiable:** the drafter's internal projection layout (tensor names/shapes) — `model.safetensors` is a 135-byte LFS pointer here and the repo has no modeling code. 12,800 = 5 taps × 2560 (hc-mean per completed layer) is source-verified; "what the drafter consumes beyond the taps" is card-level only.
- **DeepInfer** (`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py`): ref impl (:35-98) — per-seq `cumprod` of prev-step confidences, fresh seqs remapped to 0.0 (:57-68); global budget `extra_budget = total_q − num_seqs` (:72); stable descending argsort, ties to lower flat index (:85); each seq gets `1 + wins` where wins = its positions inside the top-extra_budget (:95-98). Kernel twin + tests in the same file (:11-32, :103-166). Confidence head (`models/mtp/dflash.py:41-79`): `sigmoid(w·[hidden ; markov_emb] / temperature[step_i])`, per-step temperature tuple from env (:62-64); confidence always on (:159).
- **Xolotl** (not re-opened; per T25-2 M4 with line refs): confidence-gated verify `while (kk < k && sig(sp_conf[kk]) ≥ conf_tau) ++kk;` (`tools/v41/engine/ds41_generate.cpp:3168-3186`), measured huge there because a rejected row costs ~25 ms of NVMe/PCIe — our marginal row is ~2.5 ms, ~10× smaller headroom.

---

## (b) Our equivalents, file:line (`/home/svg/LLM/strata-custom-2`)

| Mechanism | Basalt | Ours | Delta |
|---|---|---|---|
| Offer gate (depth scheduling) | `draft_keep` loop, `generate.cpp:9291` (+ :6717/:8402), joint ∏acc ≥ th, per-request adapt | Flat min-p on the draft's own prob: `while (T < S_mtp && dprob[T-1] >= req_spec_min_p) ++T;` `src/program/generate.cpp:15162-15164` (single-stream); pipeline readout trims `mtp.chain_prob()[j] < req_spec_min_p` :14867, :14899; pre-decide :14639; batch slots `draft_launch(..., o.spec_min_p)` :11699/:11930 | **No draft_cum anywhere** (grep: zero hits). Ours is basalt's th=0 degenerate branch, host-only |
| Per-request adapt | `DraftCumAdapt` observe/th, `generate.cpp:9365-9366` | None. Closest telemetry: depth histograms `dp_tested[d]/dp_ok[d]` `generate.cpp:15416-15417` (counts only, no per-window dprob rows); `chain_prob_[8]` buffer `include/strata/core/mtp.hpp:155-163,252`, filled `src/core/mtp.cpp:2014-2031` | The refit's input does not exist yet — a trace instrument is required (§c.3) |
| Threshold plumbing | `h_row_[2]=min_p, h_row_[3]=cum_th` mapped words, `mtp.cpp:1483-1487`; device reads via `min_p_bits` | `h_row_[0]=a; h_row_[1]=0;` only — `src/core/mtp.cpp:1646-1647` | Port adds two mapped words; row_ buffer +1 int pair |
| Draft trajectory | record_round/record_step, `mtp.cpp:1082-1135` | Same parent code: `capture_round` `src/core/mtp.cpp:1316-1364`, `capture_step` :1366-1391 (catch-up K/V + full layer on the anchor row; one row per step from previous residual+token) | Structurally identical |
| Step-end select | `mtp_step_end` (+cond tail), `verify_kernels.cu:890-929` | `mtp_select_kernel` — takes the row, writes tok_dst, mapped `out[j]`/`out_p[j]` with `__threadfence_system()`: `src/kernels/cuda/verify_kernels.cu:671-686`, launcher `mtp_select` :753-757; called at `mtp.cpp:1357` (round, j=0) and :1387 (step j) | **No conditional tail, no rule**. The port is literally the missing tail of our own select kernel |
| Launch shape | round + per-step graphs, or ONE conditional graph; PDL + prefetch branches | round + per-step graphs (`round_exec_[T]`/`step_exec_[j]` + `_c_` coupled twins); gated loop `for (j=1; j<max_steps && pj >= min_p; ++j)` with per-step launch + spin `wait_step` `mtp.cpp:1756-1782` (:1735-1743); STRATA_DRAFT_UNGATED twin (env-gated, measured −2.73 %, default off) :1716-1731/:1745-1754; ungated min_p≤0 shape :1669-1703; `mtp_select` between steps (1357/1387) | **No conditional nodes** (grep `Conditional|12040` in our verify_kernels.cu: zero hits) |
| Acceptance + commit | walk :9333-9334, `ver.commit(a+1)` :9345 | walk `generate.cpp:15384-15385`; **plus the fork's rescue** (typical/ratio, up to 2/window) :15390-15413 which advances `a` past the exact-match walk; commit `ver.commit(a+1)` :15428; emit :15473-15480; next draft `mtp.draft(...)` :15489-15490 | Rescue is ours — it changes what `observe()` should count (§c.4) |
| Rejection sampling | n/a (argmax drafts, `--coupled-draft` off by default) | `set_spec_q` before run :15364; coupled draft sampling on `_c_` graphs, `mtp.cpp:1248-1254`; contract `include/strata/core/coupled_draft.hpp:3-7` (output = f(seed), drafts only decide yield) | Closed F-235; anything offer-side stays output-exact by this contract |
| mm()/q_chain dispatch (verify side) | dense_tc etc. | `q_chain` lambda with dense_tc gate (`dtc_ok`, T24-S1) else `mm(wq,...)`: `src/core/verify.cpp:1650-1654`; `mm` sites :1418/:1434 (qkv), :1467 (gate/up), :1520 (out), :1707-1708 (k/v), :1892 (o), :2282 (lm_head per tile); mmvq_il gate :1371-1378 | Context for c_row anatomy; unchanged by any spec-lane candidate |
| Draft policy / suffix | DraftPolicy (their subset), suffix drafter | We are the donor: `DraftPolicy::chain_rate/chain/observe_chain` `src/spec/draft_policy.cpp:110-155`; `SuffixDrafter::propose_after` `src/spec/suffix_drafter.cpp:88`; dormant `Controller` with U(n) priors `include/strata/spec/controller.hpp:23-30` | Nothing to port |

---

## (c) Delta + implementation spec

### c.1 The draft_cum keep rule, derived from code (not prose)
Formulation: with `dprob[0..n)` the drafter's own probabilities for the next window's drafts and per-position acceptance model `acc(p)` (the bin table), draft j is OFFERED iff

  C_j = ∏_{i=0..j} acc(dprob[i]) ≥ th,  C_j float32, ascending-i order;

the window is `T = 1 + (number of leading drafts passing)`, capped by `S_mtp`. th = `adapt.cur`, started at `0.35·scale`, ramped toward `0.25·scale` by the request's EMA(0.97) ratio of observed accepted count to the bin model's expected count. It is a stopping rule on the JOINT survival — strictly tighter than a flat floor whenever early drafts are middling (two 0.6 drafts: flat 0.5 offers both, joint 0.43·0.43=0.185 trims at 1) — and it is exact: only what is proposed changes (`coupled_draft.hpp:3-7`; basalt's header says the same, `draft_cum.hpp:8-9`).

### c.2 Port spec (~50 lines; new header + 5 wiring edits + optional device tail)

```cpp
// include/strata/core/draft_cum.hpp — NEW (port of basalt draft_cum.hpp, MIT, headers intact)
__host__ __device__ inline float draft_cum_acc(float p) {
    return p < 0.30f ? REFIT[0] : p < 0.50f ? REFIT[1] : p < 0.70f ? REFIT[2] : p < 0.80f ? REFIT[3]
         : p < 0.90f ? REFIT[4] : p < 0.95f ? REFIT[5] : p < 0.99f ? REFIT[6] : REFIT[7]; }  // literals from refit
__host__ __device__ inline bool draft_keep(const float* probs, int j, float min_p, float th) {
    if (!(th > 0.0f)) return probs[j] >= min_p;              // th==0: today's flat gate, bit-identical
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);   // f32, fixed ascending order == device order
    return c >= th; }
constexpr float kDraftCumBase0 = 0.35f;                      // prior only; th* comes from OUR replay
inline float draft_cum_scale() { return 1.0f; }              // v1: unit — we are 87-92 % VRAM-hit, basalt's
                                                             // 1.1-3.5 MB CPU-bytes ramp does not key our placement
struct DraftCumAdapt { float base = kDraftCumBase0 * draft_cum_scale(); double obs = 0, exp = 0; float cur = base;
    float th() const { return cur; }
    void observe(const float* probs, int n, int acc) {       // rescue-inclusive `a` (see c.4); MTP windows only
        if (n <= 0) return; constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1;
        const float kLo = 0.25f * draft_cum_scale(); double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double)draft_cum_acc(probs[j]); e += c; }
        obs = obs * kDecay + acc; exp = exp * kDecay + e; if (exp < 0.5) return;
        double t = fmin(fmax((obs/exp - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float)t * (fmin(kLo, base) - base); } };
// Wiring (5 edits):
// 1. generate.cpp:15162-15164  -> `while (T < S_mtp && draft_keep(dprob.data(), T-1, req_spec_min_p, cum_adapt.th())) ++T;`
// 2. after `a` final (near :15416): if (!from_sfx && !first_window) cum_adapt.observe(dprob.data(), T-1, a);
// 3. before mtp.draft (:15490): mtp.set_cum_th(cum_adapt.th());  (new 2-line setter, mtp.hpp beside :79)
// 4. mtp.cpp:1646-1647 -> add `memcpy(&h_row_[2],&min_p,4); memcpy(&h_row_[3],&cum_th,4);` (device reads bits)
// 5. pipeline/batch trims (:14639, :14867, :14899, batch slot depth) — same substitution, per-request state
// Device tail (only with the conditional graph, §c.5): extend mtp_select_kernel (verify_kernels.cu:671-686) or append
// a 1-thread kernel: c = ∏_{i<j} draft_cum_acc(((volatile float*)out_p_hist)[i]) * draft_cum_acc(p);
// keep = (c >= __int_as_float(min_p_bits[1]));  cudaGraphSetConditional(h[0], keep); others 0.  Same order, f32.
// Guards: STRATA_DRAFT_CUM=1 default-OFF until the battery; th<=0 -> exact flat gate; observe() skips suffix/merged/
// first windows; max_steps cell clamp unchanged (mtp.cpp:1626); COHORTS/batch untouched in v1.
```

### c.3 The offline refit — produce the exact bins/features, and what it must compute
**Log (instrument, one boot, to a file — the F-235 lesson):** `STRATA_SPEC_TRACE=<path>`: per decode window one row `{i64 pos, i32 T_offered, i32 a_final, u8 from_sfx, u8 rescued, i32 spec_min_p_milli, f32 dprob[0..T_offered-1]}` (~44 B/window; dprob is already in hand at :15490's call — the variable the gate at :15164 consumed). This is the routing trace: per-window (c_row proxy = window ms already in `dt_run`, :15494) and per-depth (dprob_j, accepted_j) pairs. Alternative zero-boot source: replay nothing — F-228's stair boots used STRATA_SPEC_DEPTH histograms only, insufficient for the joint rule (they lack per-window prob vectors).

**Compute (no boot):**
1. **Bins:** edges fixed {0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99} (basalt's; keep the lattice, refit the values); `acc_b = P(depth-j draft accepted | dprob_j ∈ bin b)` over all logged (j, dprob_j, accepted_j). Our drafter is COUPLED at DT 0.85, so expect flatter/higher bins than basalt's argmax-calibrated table (their 70.1 % accepted-drafts at spec 7 vs our 74.9 %).
2. **Calibration self-consistency:** within-bin spread of acceptance (per depth × per bin). Tight spread ⇒ the bin table is a sufficient confidence model and a trained DSpark head buys nothing (T25-2 #4's go/no-go, resolved offline). Wide spread ⇒ reconsider the T14B VAT/confidence rider.
3. **Threshold replay:** emulate `draft_keep` in NUMPY FLOAT32 (not float64 — the replayed offered-width mix must match a future boot bit-for-bit) over the traces for th ∈ {0.25..0.60}; objective E[a+1]/E[T rows] vs the flat-0.5 arm on the SAME windows, priced at c_row 2.2–2.6 ms post-POOL_DIET (T14C calibrations 3.56 M1 / 4.62 conc pre-diet, `T14C-fastmtp-acceptance-math-gates.md:15,94`; break-even 0.72 expected tokens, :20). Emit th*, the width-mix table (F-230 battery input), and the 8 float32 literals.
4. **c_row(T):** bucket window ms by offered width from the same trace — the varlen/varlen-analog go/no-go for §(e)5 in one shot.

**Bit-identity obligations:** (i) float32 accumulate, ascending index order, host == device, `>=` compare — any reordering is a different rule; (ii) bins/th exported as float32 literals, keyed to pack version; (iii) draft draws are counter-keyed per cell (`put()`, `mtp.cpp:1631-1644`; coupled counter = cell+1), so trimming a chain never alters any other draft's bits — offers-only is genuinely outputs-exact; (iv) the replay must emulate the SAME accumulation as the header above or the offline th* mispredicts the boot's width mix (which, via kernel batch shapes, is the one channel that can move verify bits — basalt discloses the same class, `docs/precision/index.md:56`).

### c.4 One NEW delta basalt never faced: rescue-inclusive `a`
Our fork advances `a` past the exact-match walk via the typical/ratio rescue (`generate.cpp:15390-15413`). Basalt's `observe()` assumes `a` is the exact-match count. Decision required (spec, not code): feed `observe()` the rescue-inclusive `a` (what the window actually yielded — recommended, matches the rule's purpose) and log `rescued` in the trace so the refit can audit the bias. Unresolved in basalt by construction; flagged as a port decision, not a risk to exactness (offers-only).

### c.5 Device-conditional draft graph — applicability to OUR capture path, from docs + both codes
- Semantics confirmed (NVIDIA, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes", May 2024 + CUDA 12.4 docs): IF/WHILE nodes since CUDA 12.4 (ELSE 12.8); handle default value re-applied at each graph launch; `cudaGraphSetConditional` is called from a DEVICE kernel inside the graph; body graphs may contain kernel nodes, child graphs, memcpy/memset nodes, and nested conditionals; cooperative kernels are forbidden; `cudaStreamBeginCaptureToGraph` populates a body.
- Ours: CUDA 13 toolchain for sm_120 (CMakeLists.txt:210-211 warns on <13.0) ⇒ CUDART ≥ 12040 satisfied. Our draft path has no cooperative launches (grep over src/kernels/cuda + mtp.cpp: zero), its in-capture memcpys are D2D (`mtp.cpp:1520-1523, 1555`), and it already writes mapped host memory from captured kernels (`mtp_select_kernel`'s volatile `out[j]` — the same pattern basalt's `mtp_step_end` performs inside conditional bodies). **The port applies.** Shape = basalt's `capture_cond` (:1241-1323) over OUR `capture_round`/`capture_step` bodies: top graph [round child] → IF h_j {step-j body + cond tail}; handles default 0; launch once; host polls and breaks on the −1 sentinel (`mtp.cpp:1645` already writes the sentinels).
- What the actual const nodes are: in OUR anatomy the conditional graph's fixed prefix is the round (catch-up K/V for T cells + head layer on row a + select), and each conditional body is exactly today's `step_exec_[j]` contents (record_forward(1,row) + `mtp_select`) plus the ≤8-instruction cond tail. Nothing else in the window is re-captured: verify/commit graphs stay separate and untouched — this is a DRAFT-loop restructure only (the same scope F-229's twin had).
- Re-pricing the +0.5–1.5 % claim: F-229 measured the launch-shape-only twin at −2.73 %, of which ~0.19 ms/window was wasted-step GPU tail (gone under conditionals) and ~0.1 ms/window the host bubble saved (kept under conditionals). Ceiling ≈ +0.9 % M1 (0.1 ms on an ~11.5 ms window); the host still polls each produced draft (PLE prefetch + readout stay, :1657-1666/:1773), and the round launch + end-of-draft `cudaStreamSynchronize` (:1785-1788) remain. Honest band: **+0.3–0.9 % M1**, conditional on g_dt.gap_ms being material (never captured, F-235 inconclusive).

### c.6 DFlash2 multi-tap → what our T14B/T14C pipeline must produce
Verified formulation (§a.6): taps = the hc-mean (mean over the 4 hyper-connection streams) of the COMPLETED output of zero-based layers [5, 15, 25, 35, 45], 5×2560 = 12,800 features, captured at TP1/CP1, asserted by the SGLang patch. DFlash2's drafter consumes these anchor-position taps (block-diffusion conditions its one-pass block draft on them); the convention CONTRAST with DSpark (ENTERING-layer taps) stands as recorded in T25-2 M5. For OUR head — whose input today is the anchor's 4-stream R (HC×N) through `hyper_connection_mixer.hc_norm` + `input_mix_weight_down/up` (`mtp.cpp:1199-1223`) plus the embedding — the port is a SEED-side input widening: taps enter only at the anchor (steps j>0 recurse on the drafter's own hidden, as today). Availability is real: our verify pass computes every layer of every committed row each window, so the anchor row's five hc-means exist on-device at draft time; the plumbing analog is maya's ds4 `seed()` (T25-2 M5). The pipeline currently lacks:
1. **Engine tap capture:** the verify/session graph materializes ONE residual (overwritten per layer); capturing 5 hc-means per committed row = k extra device buffers + k hc-mean reductions per row inside the captured graph (serve-side, M/L; graph-shape change).
2. **Harvest record v2:** `--dump-final-r` writes `{i64 pos, i64 tok, f32[4][2560]}` = 40,976 B after each full token (`generate.cpp:16271-16276`; pinned in `T14B-exec-C-trainer-export-spec.md` §2.1). Multi-tap harvest = records carrying the 5 tap hc-means (40,976 + 5×2560×4 = 92,096 B f32), produced on the SAME prefill-walking path (taps trivially available there).
3. **Trainer + module:** exec-B's G0 module input 2×2560-equivalent → (2+5)×2560 (or 4-stream + 5 taps); T14B gates G0–G5 unchanged in form; export adds 5 hc-mean/projection tensors to the pack.
4. **Pack contract v2:** the 31-tensor mtp/rt SHA contract (F-51/52/87/100) breaks deliberately — version the manifest, keep stock `experts.bin`/`draft_vocab.bin` byte-identical (F-114 pattern).
5. **Serve-side anchor-tap plumbing** (item 1's decode counterpart) — the only genuinely new engine code.

Band honesty: +2–4 % M1 ON TOP of T14B is a CLASS estimate (T25-2), not a transfer: DFlash2's own AL delta over stock MTP is +0.18/+0.42/+0.33 (their card) — BELOW our T14B target of +0.55–0.7 over banked 2.78 (T23-dflash2 verdict stands; this is NOT a T14B skip); the 77.1 % first-draft / separable-quartiles existence proof is DSpark's, a different drafter on a different model. Our F-235 deep-position deficit is positional; taps buy content conditioning. The two compose. GATE: train an offline probe head on harvested taps and read Δ acceptance on the G0.2 stair BEFORE any engine surgery (T23-dflash2 ACTION 2, unchanged).

---

## (d) Gain / cost / falsifier

| # | Candidate | Gain (our lanes) | Cost | Falsifier |
|---|---|---|---|---|
| 1 | draft_cum keep rule (refit first) | +0–1 % M1 standalone (ESTIMATE, T25-2; the flat 0.5 gate already harvests most of the mass — F-228's curve is flat-to-negative between 0.5 and 0.35); +1–3 % post-T14B when deep marginals rise (ESTIMATE). Output-exact by contract | S: ~50-line header + 5 wiring edits + the trace boot; rule default-off until the coupled-acceptance battery (F-230) | Offline: replay on STRATA_SPEC_TRACE rows at th grid; no boot to first order |
| 2 | Device-conditional draft graph | +0.3–0.9 % M1 (AMENDED from +0.5–1.5; F-229 decomposition, §c.5); zero acceptance change with th=0 (bit-identical offers) | M/L: capture_cond over our round/step bodies + cond tail in mtp_select; lazy fallback latch | STRATA_DRAFT_TIMING g_dt.gap_ms capture (built, never captured) prices the inter-step gap before any port |
| 3 | DFlash2 multi-tap (T14C+) | +2–4 % M1 class IF T14B moves first/deep marginals (LOW-MEDIUM; §c.6 honesty) | M on T14B infra (harvest v2 + trainer input); L serve-side; pack v2 | Offline probe head on harvested taps → Δ acceptance on G0.2 stair; no engine surgery first |
| 4 | DSpark confidence head (serve-side) | ~0 at our operating point: dprob IS a free per-draft confidence and the flat gate already trims on it (F-228's mechanism); a trained head pays only if the refit shows loose calibration | — (folded into #1's refit) | The refit's within-bin spread test (§c.3 step 2) |
| 5 | DeepInfer varlen budget | Single-stream 0 (degenerates to the offer gate; V(T)=9.7+2.47(T−1) ⇒ c_row flat in T on our measurements). Conc 0 to +2 % (ESTIMATE, T25-2) — DeepInfer's own table is −1.5 % at 16 seq/rank and we run 6 slots / COHORTS 2, left of their crossover; F-231 not adopted | M if ever armed | One COHORTS per-width c_row(T) boot (shared with §c.3 step 4) |
| 6 | adapt-every / placement-scale | 0 direct; feeds #1's `draft_cum_scale` (v1 = 1.0; v2 keys on measured c_row or our own CPU-expert-bytes telemetry, computed offline) | S instrument only | Zero-boot: compute our mean CPU-served expert MB/slot from expert-cache telemetry; if ≪1.1 MB, the ramp never arms — record and keep 1.0 |

---

## (e) PLAN IMPACT verdicts

| Item | Verdict | Evidence |
|---|---|---|
| **draft_cum keep rule** | **CONFIRMED, AMENDED calibration** — adopt as the W2.5/post-T14B re-stair's principled form, but (i) bins MUST be refit on OUR coupled drafter (basalt's are argmax-calibrated); (ii) th* comes from our replay, not imported 0.35 (F-228 measured two flat negatives on this door; the joint rule differs but its standalone sign on our stack is unproven — the replay decides); (iii) `draft_cum_scale()` = 1.0 in v1 (our 87–92 % VRAM-hit placement is outside their 1.1–3.5 MB CPU-bytes regime); (iv) observe() counts rescue-inclusive `a` (NEW decision, §c.4) | `draft_cum.hpp:25-76`; our gate `generate.cpp:15162-15164`; F-228 SVG-CUSTOM.md:7852-7864; F-230 width-mix law |
| **Device-conditional draft graph** | **CONFIRMED applicable, AMENDED band** — CUDA 12.4+ IF nodes apply to our capture path (toolchain OK, body restrictions satisfied, mapped-write-in-body pattern proven by our own mtp_select); scope = draft loop only; band re-priced +0.5–1.5 % → **+0.3–0.9 % M1** by F-229's measured decomposition; bit-identical offers with th=0 (M3b-compatible), F-230 battery only if combined with the keep rule | §c.5; basalt `capture_cond` mtp.cpp:1241-1323, `verify_kernels.cu:816-929`; F-229 SVG-CUSTOM.md:7866-7878 |
| **DFlash2 multi-tap (T14C+)** | **CONFIRMED as the named T14C+ rider; AMENDED band honesty** — taps [5,15,25,35,45] completed-layer hc-means, 12,800 features, verified from patch+config; NOT a T14B skip (their AL delta < our target); +2–4 % is a class estimate gated on the offline probe-head stair read; pipeline gaps = tap capture, harvest v2, trainer input, pack v2, serve anchor-tap plumbing (§c.6) | `patches/qwen38-dflash2-capture.patch:9-16,42-49`; `config.json` dflash_config; T23-dflash2 ACTION 2 |
| **DSpark confidence-schedule** | **KILLED as a separate serve-side arm** — our dprob is already the per-draft confidence and the offer gate already consumes it (F-228); a trained head adds signal ONLY if the refit's within-bin spread is wide; folded into the keep-rule refit as its step 2. The T14B VAT-rider slot stays open, unchanged | §c.3; T25-2 #4 falsifier; F-228 mechanism note |
| **DeepInfer varlen** | **KILLED single-stream; conc PARKED** — single-stream it degenerates to the offer gate and our V(T) linearity says c_row is flat in T (nothing to arbitrate); conc: we sit left of DeepInfer's own measured crossover (−1.5 % @16 seq/rank; we run 6 slots); 0–2 % conc ceiling stands from T25-2; one COHORTS per-width c_row(T) boot is the only re-opener | `schedule_mtp_varlen.py:35-98`; `inference_deployment_qa.md:243-262` (via T25-2); V(T) calibration T25-2 header |
| **adapt-every falsifier** | **AMENDED (constants + meaning)** — basalt's adapt-every is expert-TIER adaptation, not depth scheduling: `adapt_every = 4` rounds, `adapt_decay = 0.7` (0.93 serve fast path); "0.85" appears nowhere in the tree. Depth scheduling IS the keep loop (§a.2). The falsifier for the keep rule's placement scale is the zero-boot CPU-expert-bytes computation from our expert telemetry (§d #6) | `generate.cpp:495-496, 5719-5723, 9204-9253, 9343-9344` |

---

## (f) Unknowns, confidence-graded

1. **Our acc|dprob calibration on the coupled drafter** — the refit answers it; MEDIUM confidence the bins are self-consistent (our drafts sample at DT 0.85, so basalt's argmax-shaped table WILL be wrong in form, not just value). HIGH confidence the refit is runnable from one trace boot.
2. **Offered/accepted tension between instruments** — F-228 stair boots: 2.20 offered / ~2.88 tokens/window; T25-2 standing scoreboard calibration: 3.1–3.2 offered / ~2.47 accepted/window. Both measured, unreconciled (different lanes/boots). The refit must fix ONE trace source and state it. MEDIUM confidence in each within its lane.
3. **g_dt.gap_ms (inter-step GPU gap) on our box** — instrument built, never captured (F-235 inconclusive). LOW confidence in any number until one capture boot; it prices candidate #2 entirely.
4. **DFlash2 drafter internals** (projection layout consuming the 12,800 feats; training data/method) — LFS pointer + no code locally; HIGH confidence in tap/formulation, LOW in internals. Does not block the T14C+ rider (we train OUR head to OUR convention).
5. **c_row(T) per-width shape** — flat on average (V(T) linear), never bucketed per width. MEDIUM-HIGH flat; one boot decides, shared falsifier.
6. **sm_120 driver behavior for conditional nodes with mapped-host writes inside conditional bodies** — pattern proven on sibling hardware (basalt default-on), untested on our driver/toolchain combo; MEDIUM risk, mitigated by the lazy `cond_off_` latch (port basalt's fallback verbatim).
7. **Rescue-inclusive observe() bias** — new at port; MEDIUM confidence rescue-inclusive is right; the trace's `rescued` flag audits it offline before any adoption.

## Sources
Code (read this session, on-disk pins): basalt `include/basalt/core/draft_cum.hpp` (whole), `src/core/mtp.cpp` (:1082-1323, :1459-1608), `src/kernels/cuda/verify_kernels.cu` (:793-948), `src/core/verify.cpp` (:3039-3092), `src/program/generate.cpp` (:493-496, :4779, :5719-5723, :6384, :6717, :9160-9384); ours `src/core/mtp.cpp` (:1316-1391, :1600-1804, :2014-2031), `src/core/verify.cpp` (:1371-1434, :1650-1654, :2282, :1043), `src/kernels/cuda/verify_kernels.cu` (:653-757), `src/program/generate.cpp` (:11660-11930, :14467, :14639-14910, :15162-15509, :16271-16276), `src/spec/{draft_policy,suffix_drafter,controller}.cpp`, `include/strata/core/{coupled_draft,mtp}.hpp`, `include/strata/spec/controller.hpp`; DFlash2 `patches/qwen38-dflash2-capture.patch` + `config.json` (model.safetensors = LFS pointer, unreadable); DeepInfer `deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` (whole), `models/mtp/dflash.py` (:41-79, :159).
Docs/web: NVIDIA developer blog, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes" (May 2024) + CUDA 12.4 conditional-node semantics (IF/WHILE 12.4, ELSE/SWITCH 12.8; device-only set; body node whitelist; BeginCaptureToGraph; default re-applied per launch).
Campaign: SVG-CUSTOM.md F-228 (:7852-7864), F-229 (:7866-7878), F-235 (:8074-8109); T14C-fastmtp-acceptance-math-gates.md (:15, :20, :90-94); T14B-exec-C-trainer-export-spec.md (§2.1 record contract); T25-2-spec-architecture.md (standing calibration, M1-M5 sections); T23-{basalt,basalt-portstudy,dflash2,deepinfer,xolotl}.md; F-201-{mtp,acceptance-theory}; F-200-specdec.

## Methods note
All file:line anchors were read from disk this session except those explicitly attributed to T25-2/T23 files (Xolotl line refs, DeepInfer Q&A page, basalt git history, basalt `docs/precision/index.md:56`). Grep-established negatives: no `draft_cum`/`Conditional`/`CUDART 12040` symbols anywhere in our fork's spec path; no cooperative launches in our draft path; `0.85` not present as basalt's adapt decay. The DFlash2 safetensors is an LFS pointer (135 B) — no weight/tensor inspection was possible or attempted. No builds, boots, GPU work, process changes, or writes outside this file; the live server was not touched.
