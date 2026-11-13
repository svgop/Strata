# Frontier llama.cpp/ik delta audit (2026-11-13) — 595 upstream + 332 ik PRs screened vs our tree

Batch 1 of the 18-agent frontier wave. Upstream bumped to 1.0.0 this window and now ships native
Qwen3.8-Flash-Next + DSV4.1 GGUF with its own moe-cache/CPU-expert machinery — upstream is converging on OUR
hybrid concept; the moat is the window-graph + doorbell + device-plan stack. What remains stealable:

## Ranked findings
1. **Few-row MMQ (#31533) — the verify-window TC kernel we lack.** Int8-MMA across 2-24 columns; measured
   +20.6% (1.18x/step) on a 5090, Qwen3.8-27B UD-Q4_K_M spec-decode verify passes = same GPU class, model
   family, workload. Our verify path is dp4a-only (native_mmvq 0 mma; iq_kernels' only wmma is prefill).
   Honest transfer vs our already-decode-once MMVQ: price M1 +5-12% pending the A/B (the quant-wave audit's
   counterpoint stands). Lands: mmq_few.cu per-type loaders + q8_1 fragment quantizer beside q8_1_il.cuh;
   call sites verify.cpp rows 2-8 + conc batch lanes; PDL hoisting per upstream attribution. Breaks bitwise
   verify parity → KL-yardstick gate (upstream KLD <= 0.0022 precedent). MED-HIGH port.
2. **Gyro Trellis TQ2_0 (ik #2812) — the named open lane lever (fewer bytes) with a working implementation.**
   Signed-Hadamard trellis 2 bpw: vs UD-IQ4_XS at equal GPU class TG +7-12% (Qwen3.8-Flash-Next tested!).
   If MMLU >= 95 holds at ~2 bpw experts, the DRAM-ceiling lane cuts 33-50% of its bytes = the largest
   remaining M1 lever on BOTH engines. HIGH port (format+both kernels+pack regen+full battery).
3. **ClassLRU/CRF runtime recency (#31233, #31853)** — decaying per-expert counter (half-life 128
   layer-tokens) over our static heat profile: +6-7% upstream on a weaker baseline → price +1-3% here.
   Cheapest residency upgrade; the routing publishes already carry the update stream. LOW-MED.
4. **CPU-side MoE combine for DS PREFILL (ik #2901)** — ship one combined row instead of K experts per
   token: ik +20% pp on PCIe3; ~8-18% class for our DS 262K prefill on gen5; decode already owned by the
   dm fold. MED.
5. DS candidate-pool selection fused into one pass (ik #2819) — fold topk chain into index_score/topk:
   DS window −0.2-0.5 ms (pass-count, not launch-count). MED.
6. NVFP4 W4A4 first-class (upstream #31296 fused FP4 MMA pp+9.6-15.6% on 5090; ik #2868/#2880/#2892) —
   far-term ecosystem; GPU-resident dense stack first, not the lane. Park behind 1-2. HIGH.
7. moe-cache-tune.py harness steal (#30966) — sweep (slots x pool x share); upstream data shows over-filled
   cache HURTS (6.4GB: 47.6 vs 65.5-66.4 t/s at 19GB) — insure our 2248-slot standing. LOW.
8. **CORRECTNESS cherry-pick: MMQ stream-k fixup OOB write (#32067)** — vendored tree carries it (memcheck-
   blind); fix = one guard line. Trivial.
9. Boot/hygiene batch (#30679/#31321/#32043/#30542) — mostly owned already; page-touch hygiene on graph
   metadata marginally relevant. Skip.

## Negatives (banked so nobody re-prices)
Top-k MoE fusion #31288 — ours fused and richer (router+publish+ahead, T63 gates). #31391 gather — owned by
doorbell+dm fold+device plans (their win was an np8 pathology our WSTAT telemetry would show). All 5 graph
advances (#30804/#30662/#32004/#32078/#31284) — our graphs are per-layer, permanent, direct-launched;
GDN fused-only from the start. QSA kernel advances — covered/beaten (TF32-MMA fused scorer, padded strides,
no kq_mask). ik #2901 decode half — dm fold owns it. ik #2780 topk race — structurally impossible in ours.
CPU-lane kernel family (ik +25-27% class) — DEAD HERE: the lane rides 93-100% of the DRAM ceiling; only
format changes (finding 2) move it. CPU-share — we were the donor. Numerics PRs — vendored-fallback-only.
#31959 MMVF — our decode projections are custom warp-per-row GEMV already.

## Where to start
#1 (few-row MMQ) — the only same-GPU/same-model/same-workload +20.6% attached; pair with #3 (ClassLRU) as
the low-cost companion. #2 is the strategic lever the roadmap already named.

Sources: llama.cpp #31533/#31233/#31853/#31296/#30966/#32067/#32190/#31288/#31391/#30804/#30662/#32004/
#32078/#31284/#31803/#30965/#31009/#30640/#31025/#31959/#30679/#31321/#32043/#30542/#31657/#31608/#31449;
ik #2812/#2819/#2861/#2883/#2886/#2868/#2880/#2892/#2901/#2904/#2791/#2807/#2813/#2780/#2784/#2809.
