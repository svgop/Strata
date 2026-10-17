# T13A — single-stream DECODE critical path: one step end-to-end, the per-token budget, and the road to 250 t/s stable

Date: 2026-10-16 (post F-224 binary: ONEPASS planner + FUSE_HITS built, GDN_CHUNKED + DAILY + ASYNC_LAUNCH + DRAFT_TEMP 0.85 + ROUTE_RESIDENT 0.75 + TYPICAL_DEVICE standing).
Scope: M1 single-stream decode only (conc/pair touched only where the same mechanism is shared). READ-ONLY audit; no server was touched, no file outside this dossier written.
Method: line-level trace of the serial decode loop (`src/program/generate.cpp`), the verify window graph (`src/core/verify.cpp`, `src/kernels/cuda/verify_kernels.cu`), the drafter (`src/core/mtp.cpp`), the planner (`src/core/expert_source.cpp`), and the CPU pool (`src/kernels/cpu/pool.cpp`), calibrated against measured evidence: `campaign/results/standing-final-f224.txt` (M1 214.7 t/s boot), `campaign/results/diag-phase-table.txt` (STRATA_DECODE_TIMING + STRATA_VERIFY_PROFILE phase rows), `campaign/results/prof221-map.txt`, live `/metrics` (2026-10-16 ~03:28 UTC), F-222/F-223/F-224 journal entries, and the T12 audit summaries folded into `SVG-CUSTOM-TODO.md` T12 section. Everything the T12 audits already established is marked **known**; the depth below is spent on what they did not cover.

---

## 1. Summary

One M1 decode step at the all-time peak (214.7 t/s = 4.658 ms/token, `standing-final-f224.txt`) is one verify window of **~11.5 ms wall containing ~2.47 committed tokens** (server log: 220-token requests run 11.4–12.0 ms/window at 74.9–88.3 % acceptance, ~2 drafts offered/window — the min_p stair, not T=6). The window splits:

- **~9.6–10.2 ms/window (83–88 %) = the GPU's serial walk through the ~2,050-node graph** (in-graph flag spins included). Inside it: hc mixer reads ~2.9 ms (25 %), dense/gemv/attention/rec lines ~3.3 ms (29 %), VRAM expert hits ~1.73 ms (15 %, at 87–92 % of GDDR7 roofline — byte-floored, **known**), router+plan+bell ~0.83, shared+quant ~0.81, copy+combine+PCIe-slot ~0.82, head ~0.45, flag spins+gaps ~0.4–0.8.
- **~0.3–0.7 ms/window = the host pool term actually exposed** (GPU sitting in waitA/waitB/waitCPU). The full `ms_pool` accounting is larger (1.2–1.7 ms warm; 4.2–4.5 ms at prof221's 2.2 % miss mix), but most of it overlaps the shared-expert fork and the grouped VRAM matvec; F-222's T1.1 experiment measured the flagA round-trip alone at 0.26 ms.
- **~0.95–1.19 ms/window (8–10 %) = the draft phase**, and it is host-latency-shaped, not byte-shaped: a draft step moves ~75–80 MB (≈45 µs at roofline) but costs ~0.2–0.25 ms wall because the min_p gate (mtp.cpp:1697) serializes every step behind a host wake→spin→launch round-trip.
- **~0.25–0.6 ms/window = commit (async, 0.01–0.02) + emit + stage + adapt**; the Python/server detokenize tail is inside the measured wall but small (server decode_ms ≈ client wall within ~1 %).

**The path to 4.0 ms/token (250 t/s)** needs −0.66 ms/token = **−1.63 ms/window at fixed acceptance, or +17 % tokens/window, or a mix**. The audit's conclusion: the micro-fix tranche (hc norm-fold, draft un-gating, FUSE_HITS+PCIe-skip+router fusion, barrier diet, head IL-at-T=1, sampler single-sync) sums to a plausible −0.9..−1.7 ms/window — enough to *touch* 250 in good boots — but **the only single lever comfortably above the line is re-lengthening the offered draft chain** (the min_p stair currently stops the chain at ~2 drafts; every extra accepted row costs ~0.55 ms of window and returns a full token), which is one config knob (`--spec-min-p`) plus an existing diagnostic (`STRATA_SPEC_DEPTH=1`), and it is output-changing (M3b + 2× MMLU-Pro gate class).

---

## 2. The per-token budget map (standing M1, file:line evidence)

Window anatomy (serial loop `generate.cpp:12807-13048`; verifier `verify.cpp:2301-2564`):

| # | Segment | ms/window | ms/token | Share | Where in the code |
|---|---------|-----------|----------|-------|-------------------|
| 1 | Stage-in + PLE rows + trace clear | 0.19–0.48 | 0.08–0.19 | 2–4 % | verify.cpp:2320-2336 (`stage_inputs`, `ngram_rows` loop, mapped-fence) |
| 2 | Graph launch (async thread, adopted F-200) | ~0 critical-path | — | — | verify.cpp:2344-2357 (`HostGraphLaunch`, `STRATA_VERIFY_ASYNC_LAUNCH=1`) |
| 3 | **GPU serial window** (in-graph, incl. flag spins) | **9.6–10.2** | **3.89–4.13** | **83–88 %** | capture(): verify.cpp:1015-1890; per-class split below |
| 3a | ─ hc mixer reads (gr_read ×2/layer, unpack8) | ~2.87 | 1.16 | 25 % | verify.cpp:1497 (`gr_read_group`), fused_gr.hpp; diag stage classes `hc0/hc1 norm/down/up` |
| 3b | ─ dense lines: q8+qkv gemv, conv, ab, z, rec, out-proj, QSA attn/q | ~3.3 | 1.34 | 29 % | verify.cpp:1188-1491; diag classes `gemv/z/rec/ab/conv/out-proj/attention/q+q-idx` |
| 3c | ─ VRAM expert hits (grouped matvec) | ~1.73 | 0.70 | 15 % | verify.cpp:1665-1676, 1696 (`native_expert_grouped`); 87–92 % roofline (**known**, T12) |
| 3d | ─ router+plan+bell (+route_resident) | ~0.83 | 0.34 | 7 % | verify.cpp:1519-1557 (router GEMV, top10, `native_route_resident`), 1558-1590 (`resident_plan`, `doorbell_publish_res`) |
| 3e | ─ shared expert + quant | ~0.81 | 0.33 | 7 % | verify.cpp:1597-1635 (`shared_expert_multi`, late fork 1499-1511) |
| 3f | ─ copy+combine + PCIe slot nodes | ~0.82 | 0.33 | 7 % | verify.cpp:1700-1736 (`fetch_blobs` slot, PCIe `grouped` **always launched** 1709, `copy_rows*` 1711-1730, `moe_hit_add` 1736), combine 1743-1763 |
| 3g | ─ head: output mixer + 248,320×2560 Q6_K mmvq + greedy sample | ~0.45 | 0.18 | 4 % | verify.cpp:1794-1886 (mmvq at 1862-1868; `native_mmvq_il` gated T≥2 at 1863) |
| 3h | ─ waitA/waitB/waitCPU flag spins + gaps | 0.4–0.8 | 0.16–0.32 | 4–7 % | verify.cpp:1685-1699 (waitA/waitB), 1714-1719 (waitCPU); spin kernels verify_kernels.cu:1487 |
| 4 | Host pool term **exposed** (GPU waiting on flags) | 0.3–0.7 | 0.12–0.28 | 3–6 % | service loop verify.cpp:2380-2470 (spin 2388, pool call 2437, flagA fallback 2446-2455, `*flag` raise 2467); planner expert_source.cpp:3694-4063 (ONEPASS stamps 3776-3811, publish fence 3941, publish 3943, fetch 3945, act-quant 3985-3993, O(m²) miss find 3996-4004, jobs fill 4005-4057); pool barriers pool.cpp:749-759, 823-852 (`run_phase` = 2×wait_parked + begin_batch + wait_done per phase, 2 phases/layer, repark unless `STRATA_POOL_DIET`) |
| 5 | Draft phase | 0.95–1.19 | 0.38–0.48 | 8–10 % | mtp.cpp:1562-1746: catchup 1583, round graph launch 1672, **min_p-gated step loop 1697-1715** (wait_step 1681-1689, launch 1700), sync 1718; per-step content = mixer+quant+drafter layer+head 1223-1227 (40,525-row Q4_0 = 58.3 MiB/step) + `coupled_draft_sample` 1240-1250 + `row_top_prob` 1260 |
| 6 | Commit (async) + emit + adapt join | 0.05–0.15 | 0.02–0.06 | ~1 % | commit graph verify.cpp:2071-2205, async 2708-2709, `wait_commit` deferred to request end generate.cpp:13051; emit printf loop 13000-13008; dt accounting 13021 |
| | **Total** | **~11.4–12.3** | **4.6–4.9** | 100 % | measured 11.4–12.0 ms/window at 2.47 tok/window (`standing-final-f224.txt` rows [2]-[7]) |

Calibration cross-checks:
- The diag warm GPU-stage tables total 11.39–12.10 ms/window (`diag-phase-table.txt` last rows) — matches the standing window length almost exactly; the standing boot's per-request hits (95.2–99.1 %) put the pool in the 1.2–1.7 ms accounting regime, of which only the flag-spin share (3h) is exposed.
- Node count: ~39/GDN + ~52/QSA + head/prologue ≈ 2,050/window (**known**, F-204 = 2,100 doorbell path); 11.5 ms / 2,050 ≈ 5.6 µs/node — the light windows are node-latency-bound (**known**, T12-GPU).
- Live `/metrics` corroboration: hit_rate 0.983–0.991 warm, pcie_share 0.001–0.017, drafts 988/1252 accepted (79 %) on the conc lane; `expert_slots 9454` (see §4 contradictions).

Light vs heavy: all numbers above are the warm/standing mixture. The prof221 heavy class (21.46 ms/window, pool 4.49) is the 2.2 %-miss/concurrent regime; its extra time is almost exactly the pool accounting + waitB tail (F-223's G2 closure: 26.06 = 18.81 GPU + 4.80 pool at 2× rows). Single-stream standing sits in the light regime.

### What must shrink to reach 4.0 ms/token

At fixed acceptance the window must lose **1.63 ms**. The byte-floored part (3c, ~1.73 ms at 90 %+ roofline) will not move without routing/bytes changes (**known**). The recoverable tranche, by segment: 3a hc (−0.4..0.8), 5 draft (−0.2..0.45), 3f+3d fusion tranche (−0.3..0.6), 4 pool exposure (−0.1..0.3), 3g head (−0.1..0.18), 3b sm_120 dense shapes (**known**, −0..0.4), 6 sampler tail (−0.03..0.08). Sum of mid-points ≈ −1.5 ms — 250 is reachable *in good boots*; making it *stable* additionally needs the acceptance/depth lever (#1 below) or the variance lane (clock-lock, **known** T8.B).

---

## 3. Ranked improvement candidates (single-stream 250 t/s stable)

Impact arithmetic convention: Δms/window ÷ 2.47 tok/window = Δms/token; new t/s = 1000/(4.658 − Δms/token). "Bit-identical" gates need only the acceptance canary; "different-bits" gates need M3b + 2× MMLU-Pro per the campaign law.

### 1. Re-lengthen the offered draft chain (the min_p stair) — the only >10 % lever
- **Mechanism**: the serial loop trims the window at the first draft whose probability drops under `--spec-min-p 0.5` (generate.cpp:12809-12812, fed by `mtp.draft(..., min_p=req_spec_min_p)` at 13017; the drafter itself stops launching steps at the same gate, mtp.cpp:1697). Measured result: only ~2 drafts offered/window (standing-final rows: 156–179 offered over ~63–89 windows; diag rows: avg T 2.4–3.3), 2.47 tokens/window. A verify window's marginal cost per extra row is ~0.5–0.6 ms (hits scale ~0.56 ms/row at roofline: 10 experts × 48 layers × 2.046 MB; the rest of the graph is nearly T-flat per F-204), while a committed row is worth ~4.0 ms. With the typical/ratio rescue already standing (TYPICAL_DEVICE=1, generate.cpp:12925-12948), any depth with marginal acceptance ≥ ~15 % is EV-positive; the FastMTP dossier (T11.R3, **known**) already concluded "250 needs the depth re-unlock".
- **Step 1 (zero code)**: A/B `--spec-min-p 0.35` (and 0.25) on the M1 lane with `STRATA_SPEC_DEPTH=1` (counter exists: generate.cpp:12114, 12952, report 13412-1414). Step 2 (S/M): an EV-aware depth policy using the measured per-depth acceptance (`policy.observe` infrastructure exists at 13038-13043).
- **Expected**: depths 3–5 offered; if marginal acceptance is 40–60 % (typical for MTP at temp 0.85 with rescue), tokens/window 2.47 → 3.1–3.6 at window 12.3–13.5 ms → **4.0–4.2 ms/token = 240–260 t/s**. If marginal acceptance <20 %, the knob is neutral-to-negative and the lane closes cheap.
- **Cost**: S (A/B) → M (policy). **Risk**: output-changing (different accepted stream on sampled traffic) → M3b canary + 2× MMLU-Pro; deeper windows raise per-window misses slightly (pool term up); watch M1 variance.
- **Confidence**: Medium (mechanism certain; the payoff hangs on the unmeasured depth-acceptance curve — that is exactly what the falsifier measures first).
- **Cheap falsification**: one boot, `STRATA_SPEC_DEPTH=1` + 3× M1 probe at `--spec-min-p 0.35`: read `spec depth: accepted/tested` per depth; if dp_ok[2]/dp_tested[2] < 0.25 the lever is dead for free.

### 2. hc mixer reads: norm-fold + bf16-native ALU variant — **known** lane, re-quantified as the #1 kernel line
- **Mechanism**: the 6 hc read classes total ~2.87 ms/window (25 % of the window) while moving only ~26 MB/layer — ALU/cvt-bound in the unpack8 path, not bandwidth (**known**, T12-MEM #2 / T12-GPU). Surviving levers per T12-GPU: fold gr_norm into down/up (−96 nodes; bitwise-checkable), bf16-native ops in fused_gr (different bits). My addition: at 2.87 ms/window this line is now *bigger than the VRAM hits line* (1.73) in the warm M1 regime — it deserves the top kernel-engineering slot for M1, not just conc.
- **Expected**: −0.4..0.8 ms/window → **+7..+14 t/s**. **Cost**: M. **Risk**: norm-fold bitwise-checkable (canary only); bf16-native = different bits → full gate.
- **Confidence**: Medium. **Cheap falsification**: STRATA_VERIFY_PROFILE=1 one probe — if `hc0/hc1 down` classes don't move when the fold arm is on, the norm share (<0.35 ms) is the whole story and bf16-native is the only path.

### 3. Draft phase: un-gated full-chain launch (kill the per-step host round-trips) — NEW
- **Mechanism**: in the gated path each step is launched only after the host has spun on the previous step's mapped output (mtp.cpp:1697-1715: `wait_step(j)` → `cudaGraphLaunch(step_exec_[j+1])` → `cudaStreamQuery` → `prefetch_ple`), so every step pays a host wake+launch bubble (~50–100 µs) on top of ~45–60 µs of GPU work (drafter layer 13.8 MB Q2_0 + Q4_0 head 58.3 MiB + mixer). The **same step graphs already exist un-gated**: the min_p≤0 twin (mtp.cpp:1631-1665) enqueues round + all steps back-to-back and still reads per-step `h_out_`/`h_prob_` (1653-1654). Because the graphs and Philox draws are identical, drafts and dprob are bit-identical; the host-side T-trim (generate.cpp:12809) discards over-depth drafts exactly as today. Extra cost: ≤2 discarded steps × ~0.05 ms GPU when the stair cuts.
- **Change**: when standing (coupled, `req_spec_min_p>0`), launch all `max_steps` graphs immediately, then run the existing readout loop. ~15 lines around mtp.cpp:1666-1716.
- **Expected**: draft 1.0–1.2 → ~0.6–0.8 ms/window → **−0.2..0.45 ms/window = +4..+9 t/s**. **Cost**: S. **Risk**: LOW; bits of the committed text identical (same kernels/seeds) → acceptance canary only.
- **Confidence**: High on mechanism, Medium on size (the g_dt instrument has never been run in standing — no `strata mtp draft timing` line exists anywhere in the repo logs).
- **Cheap falsification**: `STRATA_DRAFT_TIMING=1` for one 300-round block: if `host gaps` + `waits for the steps` ≥ 0.3 ms/round the win is real; if the round graph + steps alone account for ~1 ms, nothing to win.

### 4. The always-on empty nodes: FUSE_HITS arm + PCIe-slot skip + router/plan/bell fusion — **known**, with new line-level confirmation
- **Mechanism**: per layer, post() launches the PCIe grouped-expert node (verify.cpp:1709) and its waitB (1698) even when `counts[2]==0` (live pcie_share 0.001–0.017 — the slot is empty ~every layer), plus `moe_hit_add` (1736) whenever `hits_in_combine` is false; `hits_in_combine` (1733-1735) is the FUSE_HITS fold that already exists and was only recently un-hard-falsed on CUDA (de05c182, **known**). Router chain = GEMV + top10 + route_resident + doorbell_publish_res + resident_plan ≈ 5 nodes/layer (1519-1590). T12-GPU ranked exactly this tranche (~150 nodes + the combine fold).
- **Expected**: −0.3..0.6 ms/window → **+6..+12 t/s**. **Cost**: S for the FUSE_HITS arm (env exists) + the empty-PCIe skip (a capture-time conditional or counts-baked grid); M for the router fusion. **Risk**: FUSE_HITS claimed bitwise (ZADD fold, native_moe.cu:46 — verify on our stack); new fusions bitwise-checkable.
- **Confidence**: Medium. **Cheap falsification**: `STRATA_FUSE_HITS=1` ABAB; the stage table's `copy+combine` class must drop ~0.2–0.3 ms with hits_added on.

### 5. Pool barrier diet: STRATA_POOL_DIET + hot workers — **known** (T12-HOST #2), sized down for M1-warm
- **Mechanism**: every layer with ≥1 CPU miss runs 2 phases × (wait_parked + begin_batch + wait_done + repark) (pool.cpp:749-759, 904); with `STRATA_POOL_SPIN_US=100` (launcher) workers futex-sleep between layers, so each phase re-wakes them. The diet arm skips the repark (epoch-claim safety argument in-file at 755-758, 900-903). At standing M1 (hits 98–99 %) only ~12–25 miss experts/window exist, so the exposure cap is the 0.4–0.8 ms flag-spin total (3h) — the diet recovers a fraction of it; at conc/cold it is worth 4× more.
- **Expected**: −0.1..0.3 ms/window M1 → **+2..+6 t/s** (and +2-4 % conc, **known**'s +2-4 %). **Cost**: XS (env) to S (hot-worker variant). **Risk**: LOW, in-code safety argument; bits unchanged. **Confidence**: Medium. **Cheap falsification**: one env boot; pool's own `ms_repark_`/`ms_wait_park_` accumulators (pool.cpp:909-911) print the barrier share directly.

### 6. Verify head to roofline: extend mmvq_il to T=1 + sm_120 shape for 248,320 rows — NEW
- **Mechanism**: the window head reads the **full 248,320×2560 Q6_K weight = 521.5 MB once per window** (index.txt: `output.weight ... 2560 248320`; 0.82 B/param = Q6_K) in 0.43–0.48 ms = ~1.1–1.2 TB/s ≈ 63–68 % of the 1.76 TB/s the expert-hit line demonstrates on the same card. The IL path (`native_mmvq_il`, verify.cpp:1863-1867) is gated `T>=2`; the sm_120 IL table (docs/MMVQ_IL_TABLE.md) exists. Since ~2.47 rows/window average, a large share of windows still run the plain mmvq.
- **Expected**: −0.10..0.18 ms/window → **+2..+4 t/s**. **Cost**: S–M. **Risk**: LOW–M; same kernel family with a published parity table → canary; any new shape gets the bitwise check. **Confidence**: Medium. **Cheap falsification**: standalone microbench `native_mmvq` vs `native_mmvq_il` at T=1..3 over the exact (248320, 2560, Q6_K) shape; if the delta is <0.05 ms the line is already at its practical ceiling.

### 7. Publish the q8_1 image in the doorbell (skip the fp32 activation copy) — **known** (T12-HOST #4), with a NEW scope caveat
- **Mechanism**: `doorbell_publish_res` writes the fp32 `xm` image + ids per layer over mapped PCIe (verify.cpp:1586-1590) — ~30 KB/layer at T≈3 (~1.5 MB/window) — and the pool then re-quantizes the same rows host-side (expert_source.cpp:3985-3993). Publishing the q8_1 image instead removes ~¾ of those bytes and the host quant. **New caveat found**: the shared expert (verify.cpp:1620) and the PCIe staging path still read fp32 `xm`/`h_x_`, so "publish only q8_1" is not a two-line change — the shared expert needs the q8 image too (it already accepts one: `nsw.q8_1` at 1609) or a dual publish; PCIe staging rows need care under `pcie_mode`. That moves it from T12's "S" to M.
- **Expected**: −0.1..0.3 ms/window → **+2..+6 t/s**. **Cost**: M. **Risk**: bits — CPU rows use the identical q8_1 the host would have made (bits-identical if the same quantizer runs); shared-expert switch needs its own check → canary + spot M3b. **Confidence**: Low-Medium. **Cheap falsification**: `/metrics hardware.gpu_pcie_tx_mb` delta per request before/after on a 2-window probe (the 4× byte drop is directly visible).

### 8. Sampler tail: single-sync + fold the spec H2Ds — NEW, small
- **Mechanism**: sampled requests pay, inside `Verifier::run`, a second `cudaStreamSynchronize` (verify.cpp:2537) right after the window's own (2486), plus two `cudaMemcpyAsync` H2Ds for the spec-q judgment (2525-2528) and a mapped readback (2542). The spec tensors can ride `stage_inputs`, and when `judged || sampled` the 2537 sync can replace 2486's rather than add to it.
- **Expected**: −0.03..0.08 ms/window → **+1..+2 t/s**. **Cost**: XS–S. **Risk**: LOW, bits identical (same kernels, same draws). **Confidence**: Medium. **Cheap falsification**: STRATA_DECODE_TIMING=1 `stage` column on the same prompt greedy vs sampled.

### 9. Finish PDL (intrinsics-first, selectively) — **known** (T9: half-wired, stale-read hazard)
- **Mechanism**: pdl.hpp sets the launch attribute but no `cudaGridDependencySynchronize` in the hot kernels and no programmatic edges (**known**); at ~5.6 µs/node spacing, overlapping the quant/dense chain into the previous kernels' tails attacks the largest single bucket (3b + gaps). Only a launch whose graph predecessors are all kernels gets programmatic edges (verify.cpp:935-943).
- **Expected**: −0.5..1.5 ms/window if the dense/quant chain overlaps → **+10..+30 t/s** potential, the largest *code* lever left; equally the largest blast radius. **Cost**: M/L. **Risk**: HIGH without the intrinsics (stale reads); different-schedule but same-bits → canary + stress discriminators (the F-223 class). **Confidence**: Low-Medium. **Cheap falsification**: `STRATA_DF_PDL=1` boot under nsys — if launch tails don't overlap at all today, confirm the attribute is a no-op before writing intrinsics.

### 10. Stability package for "250 *stable*": clock lock + warm-in — **known** (T8.B/F-220), included because the target says stable
- The 213–215 peak coexists with 196–204 in-band boots; the phase-map and boot logs put a meaningful share of the spread on clocks/page-cache, not code. The staged clock-lock paste + per-boot prewarm is the operator-side completion of this dossier's budget: at locked clocks the §2 map becomes boot-invariant, and the remaining variance is acceptance (draft temp/depth), which candidate #1 policies directly.
- **Cost**: XS (user sudo). **Risk**: none (measurement hygiene). **Falsifier**: `nvidia-smi -q -d CLOCK` before/after across a 3-boot ABAB.

**Explicitly retired by this audit (do not re-queue)**: T1.6's *draft-side* two-stage head — the pack already ships the draft sub-vocab "en" (40,525 rows, `data/draft_vocab_en.bin` = 162,100 B) **and** the q4 head (`--mtp-q4 all` → Q4_0 in place, mtp.cpp:246-268, boot line "draft head over 40525 tokens as Q4_0 (58.3 MiB read per step, was 81.2)"). Only the verify-side shortlist remains from T1.6, and it changes bits on the main head — deprioritized behind #1-#6.

---

## 4. Contradictions found (code/metrics vs the context facts)

1. **Expert cache slots**: context says "9,337-slot VRAM expert cache"; live `/metrics` (this audit) reports `expert_slots: 9454`, `expert_cache_mib: 18363`. The 38 % residency claim still holds (9454 × 2.046 MB ≈ 19.3 GB of ~50.3 GB pair space = 38.4 %) — the slot count is one boot staler than the context.
2. **"native_mmvq full-vocab head over ~40,525 rows"** (draft phase): the 40,525-row head is the **sub-vocab** ("en" subset of **248,320**), not full-vocab, and under the standing config it is **Q4_0** (58.3 MiB/step), not Q6_K. The *full-vocab* Q6_K head (521.5 MB) is the **verify** window's, read once per window at 0.43–0.48 ms (63–68 % roofline). This resolves T12-GPU's open "521MB?? — verify the line" item: both numbers are right, they belong to different heads.
3. **"Pool term 4.0–4.8 ms/window"**: true for the prof221 boot's miss mix (2 %+ misses) and the conc lane; the current standing M1 boot logs 98.2–99.1 % per-request hit rates and the warm diag rows show `per-layer host 1.19–1.71 ms/window`. Candidates sized off 4.5 ms/window (e.g. T12-HOST's "+6-9 %" for the plan scan) overstate M1-warm gains; the ONEPASS planner has also already landed (expert_source.cpp:3752, default-on, STRATA_PLAN_DIFF verified zero-diff per F-224 close-out commit 5040b0f6) — the 1.2–1.7 ms plan-scan saving is banked, not pending. The remaining pool levers are the barrier diet (#5) and doorbell q8_1 (#7).
4. **"Decode is flat vs context depth"** — confirmed and now mechanism-pinned: QSA top-2048 selection cap (context) plus the fact that the two depth-growing costs (KV streaming, head) are both once-per-window, not per-token.
5. No code contradiction with the doorbell/flag structure, MTP coupling (Philox = position, verify.cpp:2501-2504), or the ~2,050-node count was found; the T1.1 NEUTRAL result (flagA RTT = 0.26 ms) is consistent with the warm waitA/waitB spin totals measured in the diag tables.

---

## 5. Findings by confidence

- **High**: the window/token decomposition (11.4–12.0 ms/window, 2.47 tok/window, measured per-request in standing-final-f224.txt + diag rows); the draft-phase host-gated structure (mtp.cpp:1697-1715 vs the existing un-gated twin 1631-1665); the head geometry (248,320×2560 Q6_K = 521.5 MB verify / 40,525-row Q4_0 = 58.3 MiB/step draft, from index.txt + mtp.cpp:246-268); ONEPASS default-on and verified (expert_source.cpp:3752, F-224); PCIe-slot and hit_add nodes running every layer regardless of counts (verify.cpp:1698-1709, 1733-1736); commit async and off the critical path (verify.cpp:2708-2709, generate.cpp:13051).
- **Medium**: the hc line being the top kernel target at 2.87 ms/window; the pool exposure being 0.3–0.7 ms at standing-warm (derived: flag-spin classes + T1.1's 0.26 ms measurement); the depth/EV lever's EV arithmetic (pending STRATA_SPEC_DEPTH data); head-at-63-68 %-roofline recoverability.
- **Low**: PDL's attainable size (no overlap measurements on sm_120 exist in the repo); q8_1-doorbell sizing after the shared-expert caveat.
- **Unknown (searched, not found)**: any standing-boot capture of `strata mtp draft timing` / `strata mtp profile` (instruments exist, mtp.cpp:1529-1560, 783-829; never banked); per-depth acceptance data (counter never armed in banked runs — every M3d line in results says "not printed (arm STRATA_SPEC_DEPTH=1)").

## 6. Open questions

1. The depth-acceptance curve (blocks candidate #1) — one env boot answers it.
2. Draft phase wall split (round graph vs steps vs host gaps) in standing — STRATA_DRAFT_TIMING=1 exists; nothing banked.
3. Whether `native_mmvq` (non-IL) is actually running on the T≥2 majority (mmvq_il_supported may reject Q6_K at these shapes) — decides candidate #6's size; one microbench answers it.
4. Whether FUSE_HITS' ZADD fold is bitwise on the IQ3_S/IQ2_S/IQ3_XXS format mix on sm_120 (claimed for native_moe.cu:46; not A/B'd here).
5. Whether the shared expert can read the q8_1 image bit-economically (decides candidate #7's final scope).

## 7. Implications

- The micro-tranche (#2-#8) is worth a combined −0.9..−1.7 ms/window — it makes 250 reachable but not assured; #1 (depth) is the multiplier that makes 250 *stable*, and its first measurement costs one env boot.
- Order of execution for information yield per boot: (a) STRATA_SPEC_DEPTH=1 + spec-min-p A/B, (b) STRATA_DRAFT_TIMING=1 standing block, (c) STRATA_FUSE_HITS=1 ABAB, (d) POOL_DIET boot reading the pool's ms_ counters, (e) head microbench. All are read-only-class measurements before any bits-changing adoption.
- The conc lane shares segments 3c/3f/4/5; every bit-safe adoption here also moves conc toward its 300 target (consistent with T12-HOST's +12-24 % aggregate estimate).

## 8. Source list

Primary (code, this audit): `src/program/generate.cpp` (12807-13077, 12114, 12809-12812, 12952, 13412-13414); `src/core/verify.cpp` (1497-1763, 1779-1890, 2301-2564, 2604-2636, 2708-2711); `src/core/mtp.cpp` (246-268, 783-829, 1361-1387, 1529-1560, 1562-1746, 1174-1269); `src/core/expert_source.cpp` (3694-4069); `src/kernels/cpu/pool.cpp` (456-759, 823-912); `src/kernels/cuda/verify_kernels.cu` (1487); `include/strata/core/progress.hpp` (45-60); `/home/svg/LLM/Custom-data/packs/iq3_s/index.txt` (output.weight row), `native_experts.txt` (geometry), `data/draft_vocab_en.bin`.
Measured artifacts: `campaign/results/standing-final-f224.txt` (M1 214.7 boot, per-request window/acceptance/hit rows); `campaign/results/diag-phase-table.txt` (STRATA_DECODE_TIMING + GPU stage tables); `campaign/results/prof221-map.txt`; live `GET /metrics` on 127.0.0.1:8081 (2026-10-16).
Secondary (journal/dossiers, marked **known** where used): SVG-CUSTOM.md F-222, F-223, F-224; SVG-CUSTOM-TODO.md T12 (HOST/MEM/GPU audits), T11 (R2/R3/R4), T9, T8.B, T7.B.4, T1.6; campaign/research/F-204-node-inventory.md; F-216/F-217/F-218 journal entries via git log (24a79e78..5040b0f6).
Config: `launch-custom-iq3s-8081.sh`, `custom-iq3_s-8081.json`.

## 9. Methods note

Unproductive searches, so they are not repeated: (1) no T9-T12 dossier files exist under campaign/research/ — the T-rounds live as folded sections in SVG-CUSTOM-TODO.md and journal entries; search there first. (2) No `strata mtp draft timing` or `strata mtp profile` lines exist anywhere in the repo's logs/results — the instruments have never been banked. (3) `server-8081.out` / `custom-iq3s-*.log` contain no STRATA_DECODE_TIMING lines (the env was never set on standing boots; the diag rows came from the T12 audit's diagnostic sessions). (4) The grep for the pool file at `src/core/pool.cpp` (context's path) is wrong — it is `src/kernels/cpu/pool.cpp`. Total sources consulted: 24 files across code (9), measured artifacts (5), journal/dossiers (6), config (3), plus one live service query.
