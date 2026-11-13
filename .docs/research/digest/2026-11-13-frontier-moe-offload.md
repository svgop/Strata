# Frontier MoE host-offload audit (2026-11-13) — llama.cpp/ik/vLLM/TRT-LLM vs our MoE lane

Batch 1 of the 18-agent frontier wave. Headline: the frontier converged on OUR architecture in late 2026
(llama.cpp merged the two-tier GPU+RAM cache + CPU-compute share #30966 on Nov 7; vLLM RFC #63666 proposes our
exact pointer-table design). We lead on execution; the gaps are below. Gain classes judged against the lane's
constraint: DRAM ceiling 49.5-50 GB/s measured (F-372) — only BYTES/OVERLAP classes move it.

## Ranked findings
1. **Zero-copy host-mapped streaming for the PCIe share** — the q36 lane stages every streamed blob through a
   pinned buffer = hidden 2x DRAM round trip (read complement + write stage + DMA read stage). vLLM RFC #63666:
   in-place host reads beat gather-into-scratch by 13-23%. The pattern is PROVEN IN-TREE (ds4_store.hpp maps
   arena chunks for GPU zero-copy, 24 GB/s on RDNA4) but not wired to q36's native_expert_grouped.
   expert_source.cpp:1566-1632 (stage), :1303 (complement_registered_ off), :3631 (registration cost warning).
   MED port + MED risk (NV zero-copy read rate unproven on this box). THE bytes lever.
2. **Per-layer slot quotas uniform vs routing concentration** — layer_slot_range splits evenly
   (expert_cache.cpp:627-636); rebalance can only swap within a layer (q36_engine.cu:2294-2334). llama.cpp
   #20757 static hot/cold = +26-31.6% decode on a 5090. The weighting input (st_actual_ per-layer miss counts,
   expert_source.cpp:2900-2906) is measured and DISCARDED. LOW-MED port, bit-exact.
3. **rebalance blocks the next forward** — swap uploads end with cudaStreamSynchronize on the engine stream
   (q36_engine.cu:2322-2335) while STRATA_ADAPT_OVERLAP/LAG/NOWAIT exist one lane over (generate.cpp:1593-1634).
   LOW port, hides max_swaps x 1.93 MiB H2D behind the next window.
4. Admission ungated (any free slot, arrival order) vs #20757's SLRU 80/20 + admit-on-second-miss (+8-15pp on
   churn phases; warm lane 97-99% gains nothing). The DS4 store already has the better policy (LRU/LFU-decay)
   — q36's cache is the weaker sibling. LOW port.
5. PcieAuto tunes one scalar 3x then freezes (generate.cpp:2246-2309) vs llama.cpp's offline 2-D tuner with a
   bandwidth model; extend with drift re-open + slot-count dimension. LOW-MED.
6. Miss-to-GPU selection is routing-order tail (expert_source.cpp:4635-4677) — reorder for arena locality =
   fewer/larger DMAs (our measured 12.7 GB/s per-op floor vs 15-25 batched). LOW.
7. Prefetch copies capped at dj==0 ("too often wrong") set BEFORE the byte-budgeted io-prefetch tier existed —
   retest dj<=2 io-prefetch (never copy) against the 98.6-99.3% warm lane. Fate/AdapMoE precedent. LOW A/B.
8. DENSEGEMM dense-over-all-experts loses at T<=32 — no gap at our windows (documented no-op).
9. FluxMoE compressed-residency subsumed by GSQ quants; the bandwidth-proportional placement idea folds into
   finding 2. Thin.

## Already matching or beating
The hybrid itself (frontier merged theirs Nov 7 2026); online PcieAuto vs their offline mispredicting tuner;
overlap kit (doorbell store, zero-doorbell graph, PLAN_PARTIAL, A-CPU-over-B-GPU) vs ik's 19% sync tax;
routing-aware prefetch (depth-d confidence, PREDICT_CREDIT, budgeted io-prefetch) beyond vLLM LES; host page
class (hugetlb memfd store + mlock + paced 2MiB) vs llama.cpp's still-proposal tier; CPU lane at 93-96% of the
DRAM ceiling with no published frontier number beating it (KTransformers AMX = Intel-only); DS4 5-vs-6 gating =
AdapMoE's sensitivity gating shipped; multi-GPU DISJOINT_ADAPT (+15.5%); kernel fusion parity (router 2-launch,
LFUSE, QFUSE, QDEDUP, DEC_BATCH). Third-party scoreboard: sotaaz, same file, 12 GiB: Strata 62.9 vs llama.cpp
23.4 tok/s = 2.7x; record 215.4 t/s at 64k prefix.

## Bottom line
Three items move the ceiling-bound lane: (1) zero-copy streaming of the PCIe share (in-tree proven pattern),
(2) demand-weighted per-layer quotas (input already measured+discarded), (3) ADAPT_OVERLAP for rebalance.
4-7 cheap hardening; 8-9 no-ops.

Sources: llama.cpp #20757, PR #30966, #28248 RFC; vLLM RFC #63666, #38256; TRT-LLM DENSEGEMM blog; FluxMoE
arXiv:2604.02715; AdapMoE arXiv:2408.10284; MoE-Lightning ASPLOS'25; Pre-gated arXiv:2308.12066; Fate
arXiv:2502.12224; MoE-Infinity; ik_llama.cpp #1833; am17an blog; HF MoE offload guide; sotaaz Strata-vs-llama.cpp.
