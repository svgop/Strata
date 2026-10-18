# T15 — EASY +10% DIG: fresh code audit for low-effort wins (decode + prefill + host)

Date: 2026-10-17. Method: three read-only audits — (a) main-loop host-side read (generate.cpp reactor, mtp.cpp chain, pool.cpp, verify.cpp spin sites), (b) background agent: decode GPU-kernel side (fused_gr.cu, verify_kernels.cu, native_mmvq/moe/head, qsa_select.cu, sampler.cu, s2_expert_grouped/s_gemv), (c) background agent: prefill side (prefill.cpp, gemm.cu, kernels.cu, qsa_prompt_attn.cu, moe_fused_iq.cu, moe_mmq.cu, wmma_gemm.cu). All prices are anatomy-derived per the F-232 law (code-read before battery); none has earned a battery yet. Anchors: M1 198.5 / pair 197.6 / conc 312.4; prefill 6.0-7.0k plateau.

## HEADLINE

The easy +10% is real and it is in the **hc (hyperconnection) kernel occupancy lane**, not in the banked smalls: two grid-shape changes to the standing staged gr path (fused_gr.cu) price at **+8-9% decode bit-exact** at S+M effort. Three more S-sized env/one-line arms (norm_fast flip, LFUSE, FUSE_CONVAB) add another +3-6% if they hold. Prefill bit-exact S/M sums to ~4-5% on the 32K plateau; the single biggest prefill lever (in-tree GDN chunked WY, ~5-6%) is lossy but its perturbation (~1e-7) is three orders below the F-230 tax threshold — price it with the coupled-MTP battery, don't assume it.

## DECODE — NEW candidates (ranked)

1. **UPM_COLS 16→8 (or 4): widen the up kernel grid** — S, est +3.4-5%, BIT-EXACT.
   fused_gr.cu:320-321 (constants), :437-506 (kernel), :1175-1183 (launch). Up runs 160 blocks on 170 SMs at ~0.45 TB/s (~13-16 us of the ~30 us hc read); smem ~12 KB/block is not the limit. 320-640 blocks double+ in-flight bytes/SM. Per-row math untouched (one warp/row, same chunk order, same xor tree) → bit-identical. Coupling: gr_q8_tail (:187-197) hardcodes "two blocks per q8_1 group" (`atomicAdd==1u`); with 8 cols it is 4 blocks → `==3u`. Do NOT take gr_up_q32_kernel standalone (halves the grid — net negative here). Rider: s26_tsum.cuh transposed butterfly into warp_sum (:404-410), bitwise per :1743-1746, ~+1%. Proof: fused_gr_selftest (:2217) + gr_parity multi-vs-single + graph-replay + QFUSE byte-equality (gr_parity.cpp:416-459).
2. **82-block staged down (4-warp THREADS=128 instantiation)** — M, est +2.5-4%, BIT-EXACT.
   fused_gr.cu:864-934, :1118-1147, :1006-1038. The never-tested intermediate between 41 blocks and F-217's failed 324: keeps the per-block smem tile (xn L2 traffic ~8 MB/read ≈ 0.8 us, vs 106 MB/layer that killed F-217), doubles SM coverage 41→82, down ~150→~280-300 GB/s. Needs THREADS-templated instantiation + set_staged_attr entries + DOWN_BLOCKS inject.
   Combined #1+#2 ceiling: hc read 30→~19-21 us, 2.87→~1.9 ms/window ≈ **+8-9% M1**.
3. **Fuse gr_norm into down as a cooperative launch, weights prefetched before grid.sync()** — M, est +2-3.5%, BIT-EXACT if phase bodies verbatim.
   fused_gr.cu:715-752 + :864-934. T12-GPU lever (a) — the LAUNCH fusion, distinct from the closed arithmetic norm-fold. Down weights don't depend on xn: prefetch wv tiles while norm blocks run; the ~3.5 us norm wall disappears under the fetch. Cooperative launch legal at these grids (≤80 KB smem, all resident). Not PDL (F-191): independent bytes move early, no data-edge wait. Subsumes #4.
4. **Flip norm_fast_on() on CUDA** — S (one line + arm), est +1.3-2%, BIT-EXACT.
   fused_gr.cu:759-812 gr_norm_split_fast_kernel is WRITTEN with a bitwise-equality claim; the gate at :1069-1084 hard-returns false on CUDA ("gfx12 only until another arch is measured") — an untested gate, not a measured negative. Kills the split norm's second pass (40 KB/token L2 round-trip) + one barrier. Sequence before #3 or accept supersession.
5. **STRATA_LFUSE=1 on the 5090** — S (env-only, zero code), est +1.3-3%, bit-exact on AMD harness, CUDA canary advised.
   verify.cpp:90 (g_lfuse needs the env; no launcher sets it), :928-930 (lfuse_on; ar_on() already true at standing M1 all-resident), wiring :1517/:1622-1624, native_moe.cu:55-63. No CUDA arm exists anywhere (unlike FUSE_HITS). gfx1151 table lists it exact-by-default with measured decode gains. Caveat: bitwise proven on HIP only — CUDA compile (__fdividef, fp-contract-off) needs acceptance canary + M3b. Pairs with the queued FUSE_WAIT arm in one boot.
6. **Expose + arm STRATA_FUSE_CONVAB on CUDA** — S, est +0.9-1.7%, bit-exact by construction.
   verify.cpp:268-273 gate is hardcoded `return false` with NO env read (kernel+launcher fully built: verify_kernels.cu:156-169, :1000-1023). Mirror F-205's FUSE_WAIT treatment. Risk flag: −0.3% on gfx1201 (:267) — different card; the queue can't even run the retest until the gate exposes an env.

Examined, already tight (decode): verify_kernels.cu (bitwise-provenance notes, EXACT_T shapes, batched commits); head path (Q6_K T=1 saturated grid; mmvq_il banked; STRATA_Q6_PACKED measured STRONG NEGATIVE, SVG-CUSTOM.md:2178); expert line (87-100% roofline); qsa_select.cu (multi/wmma/tc/simt/cluster variants, good paths default-on); sampler.cu (greedy/cluster/split variants; naive single-thread path already fixed, comments :104/:279); redundant R/w_norm reads + per-row sigmoid recompute (orders below noise).

## PREFILL — NEW candidates (ranked)

Ground truth: **the rig runs the MMQ path, not fused-native** — native_supported() false on CUDA unless STRATA_PF_FUSED=1 (moe_fused_iq.cu:1131-1134), launcher doesn't set it, no boot banner. Cost split at the 6-7k plateau (32K chunk ~4.9 s): expert MMQ ~1.0-1.5 s, dense GEMMs ~1.5-2.0 s, PCIe ~1.3-1.6 s mostly overlapped, GDN recurrence ~0.5 s, combine ~0.4 s.

1. **Dedup the MMQ activation quantize: T*K rows → T rows + q8_1 gather-copy** — S-M, ~1.6-2.2%/chunk (32K), ~1% at 4K, BIT-EXACT.
   prefill.cpp:3457 quantizes T*K=327,680 gathered rows/layer (3.36 GB read of only 335 MB distinct — 10x DRAM amplification) → ~161 GB read + 25 GB write across 48 layers ≈ 105-110 ms/chunk. Quantize the T distinct rows once (ids=nullptr path exists, moe_mmq.cu:308-325), reorder with a copy16-style kernel. Verified no x-side indirection in the MMQ kernel (mmq.cuh:1038-1046; args.ids is output-row mapping only). Side effect: m.Xq shrinks ~10x (~0.5 GB VRAM freed). Proof: Xq byte-hash + bo parity.
2. **Per-group copy-stream event ops in the streamed walk** — S, ~1% at 32K, up to 4-8% where the copy stream binds (small/mid prompts), BIT-EXACT.
   prefill.cpp:2439/:2487 wait + :2448/:2496 record per ENTRY (~49k event ops/chunk); groups (MMQ_GROUP=16) already consume per-group ("the copies land in order", :3843-3845). Sibling of queued DMA_BATCH — coordinate to avoid collision.
3. **Fuse swiglu + H-quantize (write q8_1 directly)** — S-M, ~0.9-1.0%/chunk, BIT-EXACT (exact op-order replication).
   prefill.cpp:3918-3920: H fp32 write (0.84 GB/layer) then re-read to quantize; fusing saves 1.68 GB x 48 ≈ 45 ms/chunk. Prove with Hq byte-hash.
4. **(LOSSY — flag) In-tree STRATA_GDN_CHUNKED=1 WY-form recurrence** — ~5-6%/chunk, NOT bit-exact.
   kernels.cu:714-1063, already benched on this card: 1.67x @512, 1.81x @≥2048 vs standing gdn_rec_kh_kernel (:596-688). Perturbation ~1e-7 relative (:721) — three orders below the 1.8e-4 KL that cost 2.5% in F-230, so the decode tax is likely far under 2.5%, but per the F-230 law it must be PRICED with the coupled-MTP acceptance battery, not assumed. Harness: gdn_rec_parity.cu. Largest code-level prefill lever left.
5. **CUDA port of the Aurora shuffle-reduction recurrence** — M, ~2-3%/chunk, BIT-EXACT by construction.
   kernels.cu:1065-1070 documents the gfx11 variant (partials meet via wave shuffles in the old order = same bits). Standing CUDA kernel pays 2 __syncthreads per token (:659/:676) in a 64-block latency-bound loop; removing the barriers ≈ 20-30% of the recurrence ≈ 100-160 ms/chunk.
6. **Upstream #2756 (STRATA_PF_ATTN_BATCH 32→128): re-price before spending** — the flag is ABSENT here (zero hits in src/); the 32 is m.attn_batch (prefill.cpp:791) used ONLY by the qsa_decode_attn_batch fallback (:2924-2929); standing path is the whole-chunk tensor-core qsa_prompt_attn_batch (:2922). Upstream's ~9% was presumably the batched decode-kernel path — likely near-no-op on this rig.

Answered scope questions: gemm.cu dequant-scratch caching is DEAD (each dense tensor dequants once per chunk; ~8.2 GB ≈ 4.7 ms ≈ 0.1% of a 32K chunk; f16 image would cost ~11 GB VRAM). wmma_gemm.cu is dead code on CUDA (gfx11-only). moe_fused*.cu inactive on the standing recipe. "Same blob staged twice" — no (≤1 entry per layer,expert). Ring stalls — none (384 slots, issuer a ring ahead, all blobs pinned). Host grouping + per-layer sync — hidden (one layer behind, ~1 ms vs ~100 ms GPU).

## HOST-SIDE — examined, already tight

- MTP chain fully enqueued back-to-back (mtp.cpp:1949-1954); gated draft loop is the production shape and DRAFT_UNGATED closed at −2.73% (F-229).
- Pool barrier on POOL_DIET (pool.cpp:755-759); act_quant between phases serial on the calling thread (pool.cpp:807-809) but ms_multi_q is small at standing miss rates.
- The 0.4-0.8 ms "flag spin" is the GPU window's wait_flag_ge kernels idling on host-pool expert results (verify.cpp:391-412, :1691/:1699/:1719) — structural overlap per F-223, not a quick sync fix.
- Reactor busy-poll by design; cudaStreamQuery every 1024 spins at the documented ~3% cost sites (verify.cpp:2473-2476 lesson).

## BANKED-ITEM RE-PRICES (F-232 law applied)

- **EcoSpec tree filling (+12-20 banked) does NOT survive.** (i) The engine is chain-only: DraftPolicy (src/spec/draft_policy.cpp, 161 lines) picks window SIZE only; the verdict loop is a linear exact-match chain (generate.cpp:12957). Tree verification = verify-contract surgery (branching verdict + branch-point undo) — L, not M. (ii) T14C's c_row=3.56 ms/row means every tree branch row must return 0.72 accepted tokens to break even; the paper's 1.36x rides on cheap rows, which we do not have. The banked +12-20 predates T14C. Downgrade to L-cost contract change; do not battery.
- Head mmvq_il-at-T=1 (+2-4): still banked, real but small; Q6_PACKED closed negative.
- Sampler single-sync (+1-2): real but small; sampler kernels already tuned.
- Doorbell-publish fusion (+1.5-3): unchanged, M.
- ROUTE_RESIDENT margin sweep: free config sweep, still open.
- Clock lock v2: user sudo, the "stable" half.

## STACKING vs ANCHORS

Decode bit-exact: #1+#2 ≈ +8-9% → M1 ~215-217; + #4/#5/#6 (if they hold) ≈ +3-6% more → ~220-225 possible without touching bits or the queued bf16-dots lever. Prefill bit-exact: #1+#2+#3 ≈ +4-5% → ~6.8k; + GDN chunked (lossy, taxed) → ~7.1-7.2k. 9-10k still needs the lossy lane or the fused-native re-open — the +10% bucket is reachable, the 9-10k target is not reachable from bit-exact S/M alone.

## FIRST-BATTERY RECIPE (if the user green-lights)

One boot, env-class arms first (zero code): STRATA_LFUSE=1 + FUSE_WAIT + norm_fast flip (one line) — ABAB vs standing. Then the two grid-shape arms (#1 with the q8_tail counter fix, #2) behind envs, gr_parity + fused_gr_selftest before any boot. Prefill: one STRATA_PREFILL_TIMING=1 boot prices candidates 1/4/5 exactly (phases already instrumented, prefill.cpp:2000-2043) before writing code.

## SOURCE LIST
Code: fused_gr.cu, verify.cpp, verify_kernels.cu, native_moe.cu, native_head.cpp, native_mmvq.cu, arch_defaults.cpp, gr_parity.cpp, qsa_select.cu, sampler.cu, s2_expert_grouped.cu, s_gemv.cu, prefill.cpp, gemm.cu, kernels.cu, qsa_prompt_attn.cu, moe_fused_iq.cu, moe_mmq.cu, wmma_gemm.cu, mmq.cuh (third_party), generate.cpp, mtp.cpp, pool.cpp, draft_policy.cpp. Records: SVG-CUSTOM-TODO.md T12/T13 + F-217/F-226/F-227/F-229/F-230/F-232/F-233 closures, SVG-CUSTOM.md:2178, T13A/B/C dossiers, launch-custom-iq3s-8081.sh, battery-boot.sh. Two background-agent audits (decode-kernel, prefill) at 4.2M/4.6M tokens each, read-only, nothing modified.
