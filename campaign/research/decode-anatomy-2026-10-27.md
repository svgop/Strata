---
type: research
status: measured
date: 2026-10-27
lane: strata-custom-2
topic: where a decode window's time goes, kernel by kernel
---

# Decode anatomy — 2026-10-27 (RTX 5090, standing recipe, solo)

Anyone reading this later: these numbers were taken with the engine's own accounting plus an `nsys`
node-level trace. The trace perturbs absolute speed (~114 t/s against ~183 t/s untraced), so use it
for **shares**, never for absolute throughput. The engine's `STRATA_DECODE_TIMING` line is the
unperturbed cross-check.

## 1. The engine's own split (unperturbed, solo, 543-window leg)

```
14.15 ms/window = verify 12.68 (GPU-reach wait 9.36 + per-layer host 2.69 [plan .11 actq .05 jobs .13 CPU 2.24] + stage .07)
                + commit/emit 0.02 + draft 1.12
tokens/window 2.74; CPU experts 0.64 entries/layer-window; VRAM hits 31.8/layer-window; PCIe 0.08
```

- `GPU-reach wait` = the host thread spinning for the GPU to ring each layer. 66% of the window.
- `CPU` (2.24 ms) = the pool's `run_split_multi` for the ~30 missed entries per window (~74 µs each,
  consistent with the F-359/F-360 byte-bound measurement on the Q2_0 shadow store).
- `draft` (1.12 ms) = the MTP chain, serial after verify.

## 2. The GPU side (nsys `--cuda-graph-trace=node`)

**GPU is the pole at solo: 15,659 ms busy over 18,000 ms wall = 87% occupied.**

| kernel | launches | ms | % GPU | µs/launch |
|---|---:|---:|---:|---:|
| native_mmvq_il_kernel | 130214 | 2122 | 13.6 | 16.3 |
| native_gu_swiglu_kernel | 98628 | 1559 | 10.0 | 15.8 |
| dm_wait_combine_kernel | 47768 | 1391 | 8.9 | 29.1 |
| gr_down_staged_kernel | 106394 | 1345 | 8.6 | 12.7 |
| native_down_multi_kernel | 98628 | 982 | 6.3 | 10.0 |
| native_mmvq_multi_kernel | 175073 | 911 | 5.8 | 5.2 |
| gr_up75_kernel | 93886 | 870 | 5.6 | 9.3 |
| wait_flag_ge_planned_kernel | 98628 | 598 | 3.8 | 6.1 |
| attn_chunk_kernel_pre75 | 41736 | 584 | 3.7 | 14.0 |
| bf16_f32_mmvf_multi_kernel | 121413 | 488 | 3.1 | 4.0 |
| gdn_step_norm_multi | 68234 | 333 | 2.1 | 4.9 |
| route_resident_k | 49820 | 328 | 2.1 | 6.6 |
| gdn_ab_multi | 36986 | 306 | 2.0 | 8.3 |
| gdn_step_commit | 62604 | 305 | 1.9 | 4.9 |
| sampler_split_part | 5096 | 235 | 1.5 | 46.0 |

Expert math (mmvq_il + gu_swiglu + down_multi + mmvq_multi + gr_up75 + gr_down_staged + gr_up_multi)
≈ **7.9 s = 50% of GPU time**. Attention + GDN ≈ 1.5 s (10%). The head is one il launch/window.

**~2.0 s of the 15.7 s GPU-busy time is spent inside spin-wait kernels** waiting for the host:
`dm_wait_combine` (1391 ms; it carries the CPU-rows wait) + `wait_flag_ge_planned` (598 ms). Their
duration histograms are bimodal — p50 2.2 / 1.2 µs (launch floor) but p99 169 / 103 µs and maxima
9.8 / 1.26 ms. The tail is the pool being slow for a layer, not the kernel's own work.

## 3. What this rules out (do not re-explore without new evidence)

- **Per-node spacing / "too many graph nodes"**: ~2,700 nodes per window, but the wait kernels'
  p50 is the launch floor; the cost is the tail, i.e. real stalls.
- **mmvq_il table retune**: the 5090 (cc 1201) uses the RTX PRO 5000 table. Re-measuring on this
  card (`mmvq_il_parity --bench --emit-table`) yields a uniform **−4.7%** across the T=2..4 shapes
  = **~0.6% of the window**. Below the adoption bar; recorded as a shelf item, not a candidate.
- **Head kernel (Q5_K 2560x248320)**: 370 µs/window at T≈7; measured optimum is rows2/4 (363 µs) —
  already there.
- **The combine's N-parallelism**: `dm_wait_combine` already launches `dim3(n4/128, n_tok)`.

## 4. Where the next real win is

The window cost is paid **per window**, and the window produces **2.74 tokens**. Expert weights are
streamed once per window regardless of row count, so throughput is `tokens/window ÷ ms/window`:
acceptance is the only large structural multiplier left (per-draft acceptance ≈0.62 today; 0.68
would clear the 250 solo target at the current ms/window). The drafter-v3 program is the owner of
that lever. Second order: the pool's stall tail (~2 s of GPU-busy, bimodal) — the shadow store cut
its mean; the tail is what remains.

## 5. Reproducing the trace

```
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --cuda-event-trace=false -d <sec> -o /tmp/trace ./engine/strata --serve ...   # via a config whose "exe" is a wrapper
nsys export --type sqlite --output /tmp/trace.sqlite /tmp/trace.nsys-rep
```
Timestamps in `CUPTI_ACTIVITY_KIND_KERNEL` are **ns since session start**; the wall-clock offset is
`utcEpochNs - systemClockNs` from `TARGET_INFO_SESSION_START_TIME`. Without that offset the filter
silently matches zero rows.
