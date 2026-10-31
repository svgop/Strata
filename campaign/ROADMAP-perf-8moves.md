---
type: roadmap
id: strata-perf-8moves
status: active
created: 2026-10-27
owner: campaign/probe-lane.py battery + this file's exit criteria per move
---

# Strata performance roadmap — 19 moves (W1–W8 + W9–W14 + W15–W16 + W17–W19), 2026-10-27/28

Mission: ≥ +10% per lane (solo M1 / pair / concurrent 6-way) warm, more while cold, via core
engine moves only. Source: full code-only study of engine + kernels + serve stack, 2026-10-27.
Tracker board: `/home/svg/LLM/.zcode/rich-tracking/strata-perf-roadmap.board.json`.

## Baseline (fill before any engine change)

| Lane | Metric | Baseline | After-roadmap target |
|---|---|---|---|
| M1 solo warm | t/s (3-leg mean) | 207.4 (T59 262K standing @5c46e95b, lanes.csv; 131K-era F-367 bank 222.0 — the drop is the --max-context 262144 config step, not a regression) | +25–40% vs baseline |
| Pair warm | t/s (3-leg mean) | 198.4 (T59 262K standing; 131K-era 233.0) | +25–45% |
| Conc 6-way warm | rows/s + acceptance | 294.1 (T59 262K standing; 131K-era 350.8; 2026-10-27 log line: 344.9 rows/s @ 68.5% accept) | +15–30% |
| Prefill 8K / 32K | t/s (t42-prefill-ladder) | ______ (references: fresh-read 6,925 t/s at 262K, M6/M22 curves) | +10–25% |
| Depth ~134K | solo t/s / conc 6-way rows/s | 173–180 / 179–210 (2026-10-27/28 depth tests) | solo ≥ 200 / conc ≥ 260 |
| Depth 228–262K | solo t/s | 114–166 (256K verification) | ≥ 180 at 256K |
| Reuse-scan | median ms at 9K / 99K | 73 / 313 | ≤ 5 at 99K |
| Decode window anatomy | ms/window, tok/window | 11.27–11.71 ms, 2.39–2.47 tok (solo) | fewer ms/tok |

Capture commands (all from repo root):
- `campaign/probe-lane.py` — the standing 3v3 same-hour M1/pair/conc battery.
- `campaign/t42-prefill-ladder.py` — prefill ladder (8K/32K/64K).
- Telemetry: `grep -a "strata decode timing" custom-iq3_s.log | tail` (solo/pair) and
  `grep -a "strata batch:" custom-iq3_s.log | tail` (conc; contains `run/CUDA0 wait/CPU experts/
  commit/emit/adapt wait` split + `routed entries per window` + acceptance).
- `grep -a "GPU stages" custom-iq3_s.log | tail` — per-stage GPU profile
  (GDN/QSA kernel-class ms per window; batch sums both cohorts, divide ~1.9 for wall).
- Snapshot before ANY change: `cp engine/strata engine/strata.pre-<tag>` (existing convention,
  see engine/ dir). Git: repo is `main`; 2026-10-27 baseline head `50684ddd`, clean.

## Protocol (applies to every move)

1. Campaign law: every change env-gated, default off; A/B via env only until adopted.
2. Gate = 3v3 same-hour ABAB battery (probe-lane.py) on M1/pair/conc + MMLU-Pro 2x
   (`campaign/mmlu-pro-gate.py`, floor 93.3 (`campaign/mmlu-pro-gate.py 0.933`, law 3)) + mimic check (`campaign/mimic-compare.py`).
   Output-changing moves (W3 acceptance semantics, W5 dense_tc, W6 i8v3 in decode) require the
   MMLU gate to PASS both runs; bitwise-parity moves (W2, W4-THP, W7 grouping) need parity
   suites only, still run the battery for speed.
3. Rollback = env off (or binary snapshot `engine/strata.pre-<tag>`).
4. Measured context (2026-10-27 boots, RTX 5090 sm_120, IQ3_S pack, 48 layers = 36 GDN + 12 QSA,
   512 experts top-10, blob 2.05 MB, draft = 1-layer MTP + 40,525-token Q4 head, --spec 6,
   window cap 8): solo 11.3 ms/window @ 2.39–2.47 tok; GPU-reach wait 8.9 ms of it — engine is
   latency/policy-bound (~28–40% of bandwidth floor 3.8 GB/token), NOT bandwidth-bound.
   Conc: 12.14 ms/window @ 4.93 rows, CPU experts 2.09 ms exposed, VRAM hit 95.9%/CPU 3.3%/PCIe 0.8%.

## The moves (execute in this order; each is independently gated and reversible)

### W1 — Adaptive speculation (solo/pair +15–30%, conc +5–10%) — FIRST, biggest ROI
Facts: window is fixed `T = S_mtp` (generate.cpp:15917) with cut-only trims (min_p gate 15918–15921,
SpecKeep 2361–2410). A COMPLETE cost-model controller is dead code: `Controller::choose`/
`CostModel::step_ms` (src/spec/controller.cpp:40–79) referenced only by controller_test.cpp.
All inputs already banked: per-depth acceptance dp_ok/dp_tested[8] (generate.cpp:15209, filled
16187–16188), per-size EMA costs (DraftPolicy::cost_, src/spec/draft_policy.cpp:145–152, fed by
policy.observe at 16276–16281), marginal row cost shape kShape (draft_policy.cpp:19–20: 8 rows
≈3.6x a 1-row window, i.e. marginal row ≈0.4–0.6 of decode).
Do:
1. Wire Controller::choose at generate.cpp:15917 (serial) and 12422–12429 (batch): pick
   argmax_T expected_tokens(T)/cost_ms(T), T∈[1..8]; `min_gain_` guard (controller.cpp:48–56)
   emits plain T=1 decode (no draft passes) when no window beats 1/cost_ms(1).
2. Feed dp_ok/dp_tested into the controller EMAs; include the untested-position growth fix
   (controller.cpp:76–79).
3. Batch lane: per-slot depth from rolling keep[b] (already tracked, generate.cpp:12543) at
   window formation (12422–12435) instead of the global env depth (5371–5380; cap 3).
4. Phase 2 (flagged, separate gate): tree verify — runner-up rows via STRATA_MTP_TOP2
   (mtp.cpp:569–591) as depth-2+ alternatives at the SAME 8-row budget; accept longest matching
   branch; device residual sampler (sampler.cu:1016–1105) already implements the correct
   tree-accept token choice per row.
Risk: low — sizing only, model semantics unchanged; spec-min-p trim retained.

### W2 — Ungated single-graph draft chain (solo +3–6%, pair +2–4%) — hours of work
Facts: each drafted token = separate step graph launched only after host reads the previous
token from mapped memory (mtp.cpp:1699–1810); draft phase costs 0.86–1.42 ms/window (~8–9%).
STRATA_DRAFT_UNGATED (mtp.cpp:1741–1747) exists, bit-identical, code calls it "the standing
rule" (:1889) — trades ~50 µs of unexecuted steps for 2 host round trips per step saved.
Do: default it on for min_p>0; then capture round+steps as ONE graph (round_exec_[T] +
step_exec_[j], mtp.cpp:1341–1416) reading per-step tokens from mapped staging (pattern the
verify graph already uses, verify.cpp:2785). Verify via `strata decode timing` "draft" field.

### W3 — Device acceptance on batch + rescue de-D2H (pair/conc +8–20%, solo +1–3%)
Facts: serial sampled traffic verifies distribution-exact on device (spec_verify_merge_kernel,
Leviathan/Chen min(1,p/q), sampler.cu:1016–1105, zero host roundtrip). Batch lane still verifies
EXACT-MATCH on host (generate.cpp:12510–12514) — coupled drafts share the target's draw yet
disagree ~30–40% at temperature; each disagreement discards a fully-paid row. Host typical/ratio
rescue D2Hs ~248K logits up to 2x/window (generate.cpp:379–425, cap 16141–16145) while the
device kernel exists (row_typical_stats_kernel, elementwise.cu:465–512). Every temperature>0
request pays a post-window sample_tokens + full cudaStreamSynchronize (verify.cpp:4565–4574).
Do:
1. Extend set_spec_q/sample_tokens_spec (verify.cpp:3207–3236; call sites generate.cpp:16113,
   17264) to batch windows: stage per-slot q rows, judge all rows in ONE spec_verify_merge_kernel
   launch (T≤kVerifyMaxT=8 already supported).
2. Make STRATA_TYPICAL_DEVICE semantics unconditional; delete the host rescue path.
3. Fold the sampled re-sample into the graph: stage the Philox counter through mapped memory
   (window-input pattern, verify.cpp:2785); delete verify.cpp:4565–4574.
4. Batch sample_rows: one launch over S rows (verify.cpp:4238–4246 is per-row today).
Gate: MMLU + mimic (acceptance semantics change: distribution-exact replaces exact-match).

### W4 — Hide the CPU-expert rendezvous (conc +10–17%, solo/pair +3–8%)
Facts: 48–96 host-GPU handshakes per window: doorbell D2H of xm/ids/w (verify.cpp:2083–2095) →
host plan ~25–40 µs × 48 layers (measured in-code, expert_source.cpp:4602–4607) → CPU IQ3_S GEMV
for ~3% misses → GPU dm_wait_combine SPINS on SMs until the flag lands (verify.cpp:2284–2318).
CPU experts = 2.09 ms of the 12.14 ms conc window. Shadow Q2_0 store is a plain 4K-fault mmap
(shadow_experts.cpp:139–142): a cold 1.38 MB blob faults ~337 times; every adapt eviction re-arms it.
Do:
1. Deferred combine: double-buffer bo_ (consumed only by next layer's hc read), enqueue layer L's
   dm_wait_combine AFTER layer L+1's mixer/attention launches — GPU works while CPU finishes
   (verify.cpp:2232–2318; the wait kernel already carries its own guard, 2261–2262). Medium-risk
   graph restructure: check buffer lifetimes.
2. Device-driven foresight: layer L's mixed_ exists right after its router GEMV (verify.cpp:2032);
   run a cheap layer-(L+1) router GEMV + top-16 on device during L's VRAM-hit GEMV, intersect with
   the resident table, enqueue H2D on the idle copy stream. Plumbing exists: staged expert becomes
   a GPU "hit" (fs->take, expert_source.cpp:4763–4764); today fed by a CPU thread, depth-0 only,
   skips when busy (2816).
3. THP the shadow store: MADV_HUGEPAGE on the mmap (pattern at pinned.cu:291–311) or route misses
   through the multi-threaded read_direct path (expert_source.cpp:2463). Byte-identical.
Verify: conc "CUDA0 wait + CPU experts" fields in the `strata batch:` telemetry line.

### W5 — Arm STRATA_DENSE_TC for wide windows (pair +3–6%, conc +2–5%, solo deep windows +1–3%)
Facts: a persistent int8-MMA dense projection path is fully ported (dense_tc.cu) and dispatch-gated
per window width (n∈[4,8], verify.cpp:1453–1456); microbench (dense_tc_parity --bench-cases):
every covered type/shape ≥1.0x from T=4 (0.72–0.9x at T=1 — the gate excludes it). OFF only
because mask defaults 0 pending the quality canary (verify.cpp:1447–1452: output-order changes —
fixed-order fp32 scale sum, NOT parity-gated). Dense projections ≈30% of the GPU window; pair
windows run 4–8 rows constantly (2 slots × (1+depth) chains).
Do: export STRATA_DENSE_TC=63 (bits: 1 GDN qkv, 2 GDN z, 4 GDN out, 8 QSA q, 16 QSA out,
32 QSA k/v); MMLU-Pro 2x + mimic; 3v3 battery with pair/conc emphasis; if clean, snapshot
engine/strata.pre-w5 and record the default posture decision.

### W6 — Decode QSA i8v3 MMA + split-tail fix (all lanes, +3–8% at 4–32K ctx, more at 64K)
Facts: trunk decode already takes the split path (verify.cpp:1975 → qsa_decode_attn_split,
qsa_prompt_attn.cu:2275–2320) — int8-KV with fp16 hi/lo-split Q on mma.m16n8k16. Remaining wins:
(a) the i8v3 3-way byte-split true-INT8-MMA formulation is prefill-only (prompt_attn_i8v3_kernel,
qsa_prompt_attn.cu:1081–1147, dispatch 2421–2427) and measures 732 vs 233 MMA rate on sm_120 with
BETTER precision (24-bit vs fp16 hi/lo); (b) split tail waste: at saturated width 2051 cells =
65 chunks of 32 but nsp pinned at 20 (2306–2309) → ~20% of blocks launch empty and the empty-split
exit (:852) sits AFTER the full q scan+quantization (:823–850); (c) LSE merge (:1026–1041, launched
:2315) and native_qsa_gate_apply (native_qsa.cu:94–102, elementwise) are separate launches.
Do: port i8v3 into decode_attn_i8_split_kernel (:800–1022); compute nsp from device-side width and
hoist the empty-split exit above the q scan; fold merge+gate into one epilogue. Parity:
qsa_rows_bench + qsa_parity suite (precision improves). Bonus if clean: gdn_step_norm column-split
48→192 blocks (48 blocks underfill ~170 SMs at batch 1; fused_gdn.cu:21–77) + fold gdn_ab into its
prologue (fused_gdn.cu:101–129) — GDN state walk is ~216 MB/token latency-bound.
Second-pass correction: the GDN column-split already exists as `gdn_step_split_kernel`
(verify_kernels.cu:448–505) behind STRATA_GDN_SPLIT default-off — do not build new surgery;
arm and measure it as W12. The prefill-attention half of this move is deprioritized per F-218:
i8v3 prefill attention is only ~3% of the 3684 ms 8K prefill; the decode split kernel (W10)
carries the value.

### W7 — Prefill: MMQ device grouping + GDN pipelining + cuBLASLt autotune (prefill +10–25%)
Facts: MMQ path (chunks <1024 tokens) does cudaStreamSynchronize(m.cs) EVERY MoE layer before
host grouping (prefill.cpp:3741; counts 3758–3765) — full stream drain, re-serializes
attention→MoE, idles the copy overlap. GDN chunked scan serializes prep(i)→scan(i) although
prep(i+1) is state-independent (kernels.cu:1174–1215; scratch already static). Dense/hc GEMMs run
on raw cuBLASLt heuristics — tuned tables are HIP-only (gemm.cu:49–67); heuristics routinely
10–30% off for narrow-M. KV host-mirror writes are per-cell (kv_q8.cu:75–80); the block-DMA
semantics exist HIP-only (STRATA_KV_HOST_DMA, prefill.cpp:3281–3300; kv_stream.cu:276–285).
Do: emit order/bounds from a device kernel reusing fused::group (moe_fused.cu:710–726) and delete
the sync; double-buffer scratch and overlap prep(i+1) with scan(i); extend warm_gemm_libraries
(prefill.cpp:1286–1349) to benchmark top-K cublasLtMatmulAlgoGetHeuristic candidates for the ~15
fixed shapes and pin winners; port the KV block-DMA to CUDA. Battery: t42-prefill-ladder 8K/32K/64K.

### W8 — Server lane: admission overlap + PROMOTE + SSE batching (conc +5–10% + tails)
Facts: one prompt read at a time, engine-wide (ctl lock server.py:1225, _take_control 1751; engine
drains windows before reading, generate.cpp:12958–12974) — a 4K-token read blocks every other
admission's TTFT. Solo→pair join pays STOP + drain + full-KV park D2H + restore + 12–17 ms promote
(server.py:1944/1976; generate.cpp:9656+ park, 11678–11700 promote) ≈80–150 ms per join event.
One send() + json.dumps PER TOKEN PER CLIENT (server.py:6940–6943). Slot-free pickup polls at 1 s
(server.py:2038–2052). Per-token locks: _note/status_lock (server.py:4819) + Latencies.lock
(prometheus.py:62–67).
Do: round-robin chunked prompt reads between waiting admissions; add in-engine `PROMOTE <slot>`
command that runs the existing es_bg.join+es_promote+copy_to_slot sequence (BADM path,
generate.cpp:16477–16516) at the next window boundary WITHOUT STOP/park; drain slot_q per ~5–10 ms
horizon into one SSE frame (pattern exists server.py:2072–2073); claim freed slots in the BDONE
notifier instead of the poll; sample the per-token lock takes every k-th token. Battery: 6-way
conc rows/s + TTFT/ITL p50/p99 from /metrics (STRATA_SERVE_TELEMETRY=1).

## Second-pass additions (2026-10-27 re-study — W9–W14)

A second code-only pass (expert quant kernels, QSA decode tail, cold-boot graph capture, GDN
decode scan, launch-count fusion, prefill staging) surfaced six moves beyond W1–W8. Execute
order among these: W9 first (env-only, hours), then W10, W11, W12, W13, W14.

**Corrections from the campaign ledger (SVG-CUSTOM.md F-217/F-218, verified against source):**
- **HC line: grid/occupancy surgery is DEAD.** F-217 (2026-10-16) measured the wide-grid
  `gr_down_multi_wide_kernel`/`STRATA_GR_WIDE` arm (fused_gr.cu:371, gate :1529-1532) on this
  card: M1 −9.2% — the cooperative tile's xn SMEM reuse is load-bearing; breaking it trades the
  occupancy cap for an L2-traffic cap (~106 MB/layer vs 13 MB). Occupancy probe (STRATA_GR_TILEV)
  neutral. The remaining hc lever is the SGLang-style Mix/Combine FUSION (fewer launches, fused
  math — measured +7.6/+5.5% e2e on this model class) — folded into W13. F-218 law: no kernel
  surgery on the hc line without a fresh nsys per-kernel profile.
- **W4-THP sub-move superseded:** F-367 adopted the tmpfs shadow store
  (`SHADOW_EXPERTS=/dev/shm/strata-shadow-q2_0`, live in the launcher today); the 4K-fault mmap
  figure (shadow_experts.cpp:139-142) describes the old path. W4's deferred-combine and
  foresight sub-moves stand; the THP sub-move is dropped.
- **W6 prefill-attention sub-item deprioritized:** i8v3 prefill attention is only ~3% of the
  3684 ms 8K prefill — W6's value is the DECODE split kernel (W10), not prefill.
- **F-218 fresh cost map (sizing basis, ~5% profiler tax from disabled fusions):** light class
  13.35 ms/window: VRAM hits 1.93 | hc reads 2.39 | copy+combine 1.08 | gemv 0.73 |
  router+plan+bell 0.68 | shared+quant 0.64; heavy class 24.31 ms: attention 4.09 (4.4x light) |
  VRAM hits 3.46 | hc reads 4.00. Attention is the top line in heavy windows; VRAM expert hits
  are the biggest stable line across classes. Banked: campaign/results/prof218-map.txt.

### W9 — S26 expert dequant kernels on sm_120 (all lanes +6–10%, bit-identical, env-only) — CHEAPEST MOVE
Facts: the decode expert GEMV reads the IQ3_S 512-entry `iq3s_grid` LUT from device GLOBAL
memory on a dependent gather chain. The S26 kernels put the LUT in shared memory and are
BIT-IDENTICAL, but `arch_defaults.cpp:51-52` arms `STRATA_EXPERT_V2=1` + `STRATA_TSUM=1` only
on gfx1151 — they are OFF on this CUDA sm_120 box. The dispatch gate (iq_kernels.cu:4781-4792)
requires `gu_type==21 && d_type==20 && n_embd==2560 && n_ff==640` — exactly this model (IQ3_S
gate/up, IQ4_NL down, 2560/640). Target: the biggest stage lines — VRAM-hit GEMV 2.55–3.36 ms +
shared+quant 1.23–1.63 ms per window (batch sums).
Do: export STRATA_EXPERT_V2=1 STRATA_TSUM=1 in the launcher; verify bitwise parity on the
expert stage (native_gu_swiglu_kernel :2552-2618 / native_down_multi_kernel :1967-2021 vs
s26_launch_l); 3v3 battery. Fallback arm if S26 regresses: STRATA_IQ_STAGE_TABLES=1
(iq_kernels.cu:3083-3090 — stages the LUT into smem once per block for the native kernels).
Rollback: unset env.

### W10 — QSA decode attention retune bundle (all lanes +3–12% at 8K+ ctx, more at 64K)
Facts: `decode_attn_i8_split_kernel` (qsa_prompt_attn.cu:800-1022) launches grid (n_q, 2, nsp),
128 threads, ~44.7 KB smem → 2 blocks/SM. nsp code :2306-2310: `nsp = min(2*SMs/(n_q*2), 20,
chunks_cap)` → at saturated width 2051 = 65 chunks of 32, nsp=20 → splits 17-19 launch EMPTY,
split 16 gets one chunk, merge chain depth fixed at 20. The q prescale/pack is recomputed in
all 40 blocks per row (:818-843). Page lookups for streamed KV run through ONE block:
`resolve_kernel<<<1, RT, 0, stream>>>` (kv_stream.cu:230), and the 96-block copy_kernel
zero-copy PCIe reads serialize before attention each layer.
Do: nsp = ceil(width/32) (or a device-side grid guard — launch only as many z-blocks as
chunks); hoist the empty-split exit above the q scan; hoist q prescale/pack into the plan stage
(one launch per row, not per split); multi-CTA resolve (split the page table across blocks);
then port the i8v3 24-bit formulation (W6 item (a)) into the same kernel.
Estimate: 1.5–2.5× on the attention term (1.66–2.49 ms/window solo, 4.09 ms heavy-class per
F-218) → all lanes +3–12% at 8K+ ctx.

### W11 — Cold/jitter package: boot graph sweep + swap budget + spare-pool warmth (smoothness, cold +10–25%)
Facts: the log shows 2361 "captured the batch window over slots…" events — batch per-layout
graphs are captured LAZILY at round start, blocking the first round of each new layout
(~10–40 ms each); boot pre-capture covers only the serial verifier's exec_[1..8] + commit.
After idle, the adapt tier-swap cap (96/round) fires a 96-swap storm with join 2.46 ms;
promote 12–17 ms, park 24–52 ms, restore 12.4–15.6 ms; 33,046 MB file-tier reads after
drop_file_cache.
Do: boot-sweep capture of the hot batch layouts + exec_nr_ + ver_b/ver_c + MTP round/step
graphs (prewarm2.sh hook); budget adapt tier swaps after idle (decay-aware, cap per round until
the profile re-learns); keep the VMM spare pool above an es_demote watermark; prewarm2 join
replay; background fadvise/readahead of the expert shards after drop_file_cache.
Gate: cold-3v3 battery (cold boot → first-round t/s, TTFT p50/p99), not warm-only.

### W12 — GDN decode split scan default-on sm_120 (solo +3–5%, pair/conc +2–4%)
Facts: `gdn_step_split_kernel` (verify_kernels.cu:448-505) splits the 48-block GDN step into
192 blocks on ~170 SMs — the same underfill class as W6's bonus item but in the VERIFY kernel
that actually runs: `gdn_step_norm_multi_kernel` (:293-437) is 48 blocks × (128,4),
register-cached state, ≥4 barriers per token. It is gated behind STRATA_GDN_SPLIT,
default-off (:1180-1199); on gfx1151 it measured commit +8–19%, 2+ rows +3–13%, 1 row no gain,
BITWISE. STRATA_GDN_COMMIT_SPLIT is already default ON.
Do: arm STRATA_GDN_SPLIT=1 on sm_120; parity (bitwise); then fold gdn_ab_multi (:1122-1140,
48 blocks × 64 threads) into the split prologue (one fewer launch per layer). Structural
follow-up (non-bitwise, separate gate): WY-chunked decode scan for the 2+ row windows.

### W13 — Launch-count fusion: Mix/Combine + LFUSE + router/plan/bell (all lanes +3–6%)
Facts: ~288 graph nodes per window in the MoE+HC neighborhood; router+plan+bell alone is
1.43–1.90 ms. Shared expert gate/up are two separate mmvq launches unless STRATA_LFUSE
(pair-launch `native_mmvq_pair`, bitwise) is armed — shared_expert.cu:228-230. F-218: the
remaining hc lever is the SGLang-style Mix/Combine FUSION (fewer launches + fused math;
+7.6/+5.5% e2e measured on this model class). At 4.93 rows (conc), the T≥4 grouped TC path
`gemm_iq_f16_grouped_kernel` (iq_kernels.cu:5003+) amortizes the per-row GEMV shape.
Do: (1) default STRATA_LFUSE on (bitwise); (2) build the Mix/Combine-fused HC node (env-gated,
nsys-profile first per the F-218 law); (3) fuse router+top-10+plan+doorbell into one node per
layer (the doorbell is already a 1-block kernel riding the same inputs,
elementwise.cu:329-353); (4) trial grouped TC experts for T≥4 windows under the W5-style
quality canary.

### W14 — Prefill: staging-pool double buffer + MMQ host-path kill (prefill +10–25%)
Facts: extends W7. KV staging goes inline H2D per layer up to ~134 MB/layer at 131K context
(prefill.cpp:2836/3263) — no double-buffered staging pool, so the copy engine idles between
compute chunks. The MMQ host grouping path still exists even though device bounds/order emit
already exist (prefill.cpp:4597). Note: wmma_gemm.cu is gfx11-only — never used on sm_120; do
not size prefill GEMM work off it.
Do: double-buffer the KV staging pool (chunk N+1 H2D overlaps chunk N compute); delete the MMQ
host grouping path entirely (W7 item (1) then has no fallback); cross-chunk staging lookahead.
Battery: t42-prefill-ladder 8K/32K/64K.

## Third-pass additions (2026-10-28 — depth persistence & measurement stability, W15–W16)

Depth map (measured, 2026-10-27/28): 0–60K solo 209–222 / conc 341–351 · ~134K solo 173–180 /
conc 179–210 · 228–262K solo 114–166. All three degradation mechanisms verified in source
2026-10-28; the A/B verdicts elsewhere in this roadmap are same-depth comparisons and stand.

### W15 — Depth persistence (134K/256K lanes; biggest unaddressed gap)
Mechanisms (verified):
- **KV tier is static.** `--kv-resident 32768` is a boot constant (layer.cpp:575,793
  `qsa_set_kv_resident`, floor 20480). Past the tier, every QSA window pays
  `resolve_kernel<<<1, RT>>>` — atomicCAS page claims + a serial clock sweep
  (kv_stream.cu:104–160, launched :230) — then host→VRAM page copies, absent pages masked to
  −∞. At 134K × 6 conc lanes the page misses contend on the same PCIe as shadow-expert traffic
  (pcie-frac 0.35), which is why conc falls harder than solo.
- **The attention knee is the score scan.** Decode scores run the SIMT
  `block_scores_multi_kernel`, looping `max_blocks = ctx/4` blocks linearly
  (qsa_select.cu:1003,1032–1038): 8.2K blocks at 32K → 33.6K at 134K, ×12 QSA layers. A TF32
  tensor-core variant (`qsa_block_scores_tc`, sm_80+ gated, qsa_select.cu:173+) exists but only
  prefill uses it. Top-k is a single 1×1024 block over the whole score row (qsa_select.cu:2579–2580).
- **Reuse-scan is linear in depth.** `ConversationCache::best()` runs `conversation_prefix`
  (a full `std::equal`) per entry × checkpoint on the admission path
  (conversation_cache.hpp:315–330, :233–249) — the 73→313 ms median growth 9K→99K.

Sizing: KV int8 ≈ 1 KB/cell/layer → full 131K residency ≈ 1.6 GiB across the 12 QSA layers
(that is the entire pinned host mirror) vs a 19.6 GiB expert cache. MEASURE whether the stream
pools are shared or per-session before sizing the demotion (per-session × 6 lanes = 9.6 GiB).

Do (order by cost):
1. Elastic KV resident tier: resident cells scale with the deepest active conversation
   (auto-rebalance vs expert-cache slots; the adapt tier refills from the profile). Env-gated.
2. TC block scores on decode: port `qsa_block_scores_tc` into the decode multi path.
3. Prune/widen top-k: multi-CTA or two-level (pooled-of-pooled) top-k so the scan stops
   growing linearly at ctx/4.
4. Prev-window cell prefetch: adjacent tokens' QSA selections overlap — pre-DMA the previous
   window's non-resident sel_ pages on the copy stream before this window's resolve; batch KV
   page DMA across the 12 QSA layers (STRATA_DMA_BATCH-style) on conc.
5. Reuse-scan rolling hash: incremental hash per checkpoint (maintained on park/append);
   `best()` hash-compares first, full `std::equal` only on hit. Byte-identical outputs.
6. Depth battery: 134K + 256K lanes 3v3 vs the depth baselines + MMLU 2x + standard-lane
   no-regression; verify W1's controller auto-shrinks T at depth (its cost EMAs rise) — W1 is
   itself a depth-persistence mechanism and composes with every item here.

Expected (stacked, stretch): 134K solo ≥ 200, 256K solo ≥ 180, 134K conc ≥ 260. Honest
per-sub-move: 5–15% at depth.

### W16 — Measurement stability & drift discipline (protects every gate above)
Ledger facts: window drift M1 180–222 across same-boot windows under sustained load; five
back-to-back boots 02:10–03:30 confounded tonight's depth reads (final standard M1 209.6 at
03:25 was inside the banked band on the adopted binary); strata-gpu-lock.service holds clocks
and M17's ~6% deficit is a Blackwell Reliability hardware derate — not a missing lock, do not
chase it.
Do:
1. WSTAT per-depth bands: bucket window p50/p99 by active depth in STRATA_SERVE_TELEMETRY.
2. Temp/clock trace during every battery (nvidia-smi poll appended to the battery log).
3. Boot-spacing rule: interleave A/B legs across windows — never all-A then all-B back-to-back.
4. Noise-floor gate: a lane verdict must exceed the recorded drift band for that window slot.
5. Depth-slope watch: record depth-lane t/s vs depth per engine build in the ledger so depth
   regressions are caught at review time, not rediscovered.

### Board mapping (tracker rows ↔ this doc)
The tracker board (`.zcode/rich-tracking/strata-perf-roadmap.board.json`, 12-row cap)
consolidates: `w6` = §W6+§W10+§W12 (QSA decode kernel + GDN split arm) · `w7` = §W7+§W14+§W19
(prefill) · `w4` = §W4 + §W13 items (2)-(3) (MoE/HC neighborhood: defer + fusion) ·
`w5` = §W5 + §W13 item (4) (TC quality-canary arms) · `w9` = §W9 + §W13 item (1) (env-only
arms) · `w11` = §W11 · `w15` = §W15 · `w0` carries §W16's measurement discipline ·
§W17 (cadence flag) + §W18 (batch draft-depth ramp + telemetry gap) live as items on `w8` and
`w1` respectively. Doc section numbers are canonical; row ids are stable.
Numbering note: `MEASURED-opt-map-2026-10-27.md` proposed its own W15/W16/W17 — they are §W17,
§W18, §W19 here (this file's W15/W16 are depth persistence and measurement stability).
SCOUT corpus (2026-10-31): one research digest per open row, 12 files at
`.docs/research/digest/` (w0..w15), each 3-6 reference implementations, confidence-graded,
folded into the board rows. Notable adjudication: the w9 digest's "S26 already armed on CUDA"
claim was refuted against source (arch guard arch_defaults.cpp:32; nv_defaults.cpp has no
sm_120 table) — the arm items stand.
TRIAGE corpus (2026-10-31, law 15): every board item priced KEEP/DOWNGRADE/KILL in
`.docs/research/digest/triage-t1..t6*.md`; verdicts are tagged on the board items themselves.

Order of attack (TRIAGE-corrected 2026-10-31; supersedes the measured-map order above):
0. FALSIFIER PACK (no legs, ~a day): W0 gate-quality half-day (drift floor + depth
   re-measures + WSTAT bands — every ±4% verdict needs it) · W3.1's one instrumented conc
   window (prices the +24-34% needle's mass) · W1's rows/window + per-depth dp dump (bimodal
   verdict) · W7's one diagnostic boot (STRATA_PREFILL_TIMING + KV_PREFETCH toggles; prices
   items 1/5/6/9) · W9's ~90-min nsys of the VRAM-hit GEMV · W15's GPU-stages scan check.
1. W2 (bit-identical promotion, +3.9-6.6% solo/pair) — first ladder leg.
2. W3 bundle (biggest needle: +24-34% tokens/window) — BEFORE W1.
3. W4.1+W4.2 one candidate (conc wall time) → nsys → W4.3+W4.4.
4. W9 composed env leg + W6's GDN split arm (both bitwise) in one promotion.
5. W11 warmth; W8 PROMOTE + cadence (N=2 + saturation guard); W15 reuse hash (row champion);
   then the rest by the row orders.
NOT RUN (law 15 — killed or must compose first): SSE batching, per-token lock sampling,
standalone top-k prune, standalone W3.2-4, W7 autotune or staging alone, W5 dense_tc unless
the parity multiplier reads ≥1.2× at T=4-8, W15 tier without a paired demotion.


## Fourth pass (2026-10-28 — measured-map integration, W17–W19)

Source: `campaign/MEASURED-opt-map-2026-10-27.md` (live self-telemetry pass: /metrics, 450-sample
series, SSE traces, 1/2/4/6-lane probes, cost-model fit, 80k→108k knee series, 11,373-block log
mining). Its proposed "W15/W16/W17" candidates are recorded here as **W17/W18/W19** to avoid
colliding with this roadmap's W15 (depth persistence) and W16 (measurement stability). The map
itself is frozen evidence; this file is the plan.

### Measured corrections applied to existing moves
- **W1 gets a measured number (M15):** window wall = 8.4 + 1.57×rows ms (R²=0.99,
  context-independent ≤~72k). Solo depth-8 windows run 30–41% row-efficient; trimming to ~4 rows
  keeps the tokens at ~16.3 ms/window → **+35–38% solo, measured**. Batch D=1 lanes run 78–91%
  row-efficient — the D=2 upside is small (see W18).
- **W8's SSE batching is latency-neutral (M14):** client gap ≈ server gap on loopback; the
  per-token send loop adds nothing measurable. Keep the item only as a GIL/throughput note.
- **W3 must not re-introduce the F-30b rescue (M23):** STRATA_BATCH_RESCUE=0 is law — the batch-site
  typical rescue cost MMLU 95.7 → 91.4–92.9 (F-30b). W3's p/q rejection sampling is
  distribution-exact (unbiased) and different machinery; the MMLU gate stays mandatory.
- **W11's costs are size-scaled, not constant (M19/M23/M25):** demote parks 21.6–30.6 ms on small
  conversations but up to 299 ms (park 214.6) on large ones; reuse-scan median 73→313 ms from
  0–9k to 90–99k ctx with a 2–413 ms variance band; conversation-cache eviction costs a FULL cold
  re-read on the first turn after (11.3 s at 80k, M25). Conversation cache ran 24/24 slots with 7
  evictions — budget sizing is part of W11. Elastic promote is cheap (1.1–6.6 ms) — the cost is
  all on the park side.
- **W15 depth gets knee + reuse-granularity facts (M20/M21/M16):** windows pick up a
  +4.7–7.6 ms context term above ~80k (35% of the window) with p99 widening; cross-conversation
  reuse caps at kv-resident 32768 and, once the cache evicts, falls back to whole 32768-token
  blocks (observed 65536/98304 landings) — up to 32,768 tail tokens re-read at bulk rate.
- **Clock law (M17/M24):** chronic ~6% SM deficit is the always-on 0x400 "Reliability" flag — a
  Blackwell hardware V-F guardband, root-only, NOT a software lever (lock service already runs).
  Episodic power-cap spikes (1785 MHz transient) occur during concurrent cold-prefill storms —
  W17's cadence flattens that condition.
- **Prefill shape quantified (M6/M9/M22):** ~1.0–1.3 s fixed launch tax + ~6.8k tok/s marginal
  ceiling (PCIe-capped, 31 GB/s spike at 475 W); small reads <2k run 1.2–2.2k tok/s launch-bound —
  4–5× below bulk. The W19 prize is the fixed term.
- **Lane curve baseline (M8):** 1/2/4/6 lanes = 209 / 164 / 279 / 341.5 aggregate tok/s, window
  p50 17.1–23.8 ms, rows/window 3.5–9.6 — the GPU absorbs rows almost free. Batch drafts are
  counted as verifier rows, NOT in drafts_offered (telemetry gap, fixed under W18).

### W17 — Enable prefill cadence (boot flag; conc smoothness)
`STRATA_PREFILL_CADENCE=1` (+ `PREFILL_CADENCE_N`, default 2) is shipped, documented, and OFF
(server.py:325–339, cites vLLM #44558: P50 ITL 274.9 → 32.1 ms). M18 measured the disease on
this engine: cold 4k/16k reads interleaved with six decoding lanes spike per-lane ITL to
0.1–0.8 s and inflate the injected read's TTFT 3–5× (6.6–6.9 s) — one prompt-chunk read forces
the decode graph out of replay. Zero code change, one boot. Measure TTFT p50/p99 alongside lane
ITL (cadence delays prefill admission while decode is busy). Pairs with #656 BYIELD (fired 0
times this session — the long-prompt case is the live one).

### W18 — Batch draft-depth ramp + telemetry gap (env-only, one measurement)
`STRATA_BATCH_DRAFT_DEPTH` is 1 (generate.cpp:5375); batch_mtp is live with 6 slot drafters.
M8/M15: the batch lane is already 78–91% row-efficient at D=1 — D=2 adds rows the lane can
barely convert, so run the ramp (D=2, D=3; `batch-draft-depth-notes.md` has the test) for the
measurement, do not plan on the gain. Alongside, fix the telemetry gap: batch drafts are counted
as verifier rows and never appear in `drafts_offered` (the 5,462 "drafts accepted 0 of 0" lines)
— make batch acceptance directly readable per request.

### W19 — Small-read prefill fast path (agent-loop latency)
M9/M22: reads of ~1–2k tokens run at 1,207–2,236 tok/s — launch-bound, 4–5× below the 6,512
tok/s bulk rate; the curve is ~1.0–1.3 s fixed + 6.8k marginal, so a 2k read costs 1.03 s where
only ~0.3 s is compute. A per-window small-read path that skips the full prefill staging
machinery below ~2k tokens cuts the agent-loop marginal term. OPEN ONLY AFTER W8/W11 land; the
acceptance metric is the per-request prompt_ms term in the `requests` block (visible today).

## Flagged experiments (NOT adopted recommendations)
- Re-A/B STRATA_COHORTS=2 vs STRATA_BATCH_G2 now that STRATA_DM folded the doorbell waits —
  cohorts duplicate every dense-weight read (generate.cpp:9366–9395).
- Re-arm STRATA_DF_PDL=1 on sm_120 (it was neutral when MoE waits dominated; what remains
  intra-graph is kernel-boundary latency on short quantize/GEMV chains).

## Rollback registry (fill as moves land)
| Move | Env / binary rollback | Gate result | Date |
|---|---|---|---|
| W1 | controller off-env / engine/strata.pre-w1 | | |
| W2 | STRATA_DRAFT_UNGATED=0 / pre-w2 | | |
| W3 | STRATA_SPEC_PROB=0 + rescue path restore / pre-w3 | | |
| W4 | deferred-combine env off / pre-w4 (THP sub-move dropped — F-367 tmpfs shadow store live) | | |
| W5 | STRATA_DENSE_TC=0 (or unset) / pre-w5 | | |
| W6 | split-path fallback (env) / pre-w6 | | |
| W7 | grouping env off / pre-w7 | | |
| W8 | server flags off / pre-w8 | | |
| W9 | STRATA_EXPERT_V2/STRATA_TSUM unset (env-only, no binary change) | | |
| W10 | nsp/retune env off / pre-w10 | | |
| W11 | prewarm2 hook off / swap-budget env off / pre-w11 | | |
| W12 | STRATA_GDN_SPLIT=0 (default-off today) | | |
| W13 | STRATA_LFUSE unset; mix/combine fusion env off / pre-w13 | | |
| W14 | staging-pool env off / pre-w14 | | |
| W15 | kv-resident rebalance env off (static 32768 restored) / pre-w15 | | |
| W16 | telemetry/battery protocol only — no engine change | | |
| W17 | STRATA_PREFILL_CADENCE unset (off) — env-only | | |
| W18 | STRATA_BATCH_DRAFT_DEPTH back to 1 (default) — env-only; telemetry fix is additive | | |
| W19 | small-read fast path env off / pre-w19 | | |
