# F-204 scan: decode-window NODE INVENTORY (agent, 2026-10-15)

**Measured config (--native, --kv int8, dma/doorbell path, VERIFY_DEVICE_PLAN, ROUTE_RESIDENT 0.75, PLE_BATCH):**

- Window totals at T=8 single-session: GDN layer = **39 nodes** (doorbell) / 28 (all-resident);
  QSA layer = **52/41**; window ≈ **2,100 nodes** (doorbell) / ~1,580 (all-resident) / measured
  ~1,800 = the resident/doorbell mix.  **BATCH windows at S=6: ~49/GDN + 91/QSA ≈ ~2,900 nodes**
  (per-row QSA suite + per-slot conv/recurrence loops).
- T-scaling: nearly flat (multi-row launches dominate); only fp16-KV append, the PLE row loop,
  and q8_1 interleave (T∈[2,4]) multiply with T.

**TOP-10 FUSION LIST (node deltas/window; full table in the agent output, key file:line in verify.cpp):**
1. Fold q8_1 quantize into producers (×4-5/layer; hooks exist: gdn_step_norm_multi xq_out, fused_gr q8_mixed; behind QFUSE-class gates) — **−180**.  CAUTION: F-187's QFUSE single-form test was a mirror trade; re-test per-form.
2. Shared-expert chain 6→3 via **STRATA_LFUSE** (implemented + bitwise-checked; master switch g_lfuse, verify.cpp:90) — **−144**.
3. **Fused wait+copy doorbell forms** (wait_copy_i32_from_mapped / copy_rows_from_mapped_after exist; fuse_wait_on() hard-false on CUDA → F-205 un-gated behind STRATA_FUSE_WAIT) — **−144**.
4. Router chain 4→2 (top-10 into gemv epilogue; route_resident reads same logits again) — **−96**.
5. hc read 3→2 (gr_norm folded into gr_down; down→up stays split through lo_) — **−96**.
6. **Batch-window QSA per-row suite → per-row-pool batched kernels** (array of pool pointers, gridDim.y=S; the single-session _batch/_steps kernels prove row-wise arithmetic) — **−330 at S=6**; batch GDN conv/rec loops same way (−20).
7. PLE row loop → native_ple_postops_batch_snap (exists, bitwise-equal, STRATA_DF_PLE) — **−56**.
8. QSA small-pairs: K+V mmvq one launch; attn chunk+merge one kernel for decode-sized n_chunks; kv_resolve 2→1; tail-snapshot copy into indexer append — **−48**.
9. Expert-side: rebase into fetch_blobs; skip PCIe grouped-expert node when counts==0 knowable; hit_add into combine (native_moe_combine_multi_hits_added exists); gr_write into combine — **−100..−140**.
10. Miscellany: expert-input quantize into hc-ffn q8 tail; gate-apply writes q8_1; prologue copies via copy_from_mapped_multi; head mixer 3→2 — **−60..−90**.

**Sum ≈ −900..−1,000 nodes from ~2,100 (→ ~23-25/layer, the TRT-LLM band).  Cheap tranche = #2+#3+#7 (gates); then #1/#5 (new kernels); #6 is the batch-window prize.**
