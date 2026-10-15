# F-204 scan: the emit/slot-draft path — full map + R33 batching plan (agent, 2026-10-15)

**Corrections to priors:** draft vocab subset = **40,525 tokens** (rt/draft_vocab.bin 162,100 B), not 4.6k; window row cap kVerifyMaxT=8 → **4 MTP slots draft per window** (2 rows each) of the 6.

**Timeline (per window, generate.cpp:10135 batch_step):** verify (w0) → acceptance loop + commit_slot_prefixes (F-196 async) → EMIT (bt_emit, 1.87ms): per active slot — nonfinite check, token emit + printf, 80KB residual D2D on the drafter's stream, `slot_mtp[b]->draft(2,...)` which is SYNCHRONOUS (launch → spin on h_out_[0] → cudaStreamSynchronize, mtp.cpp:1666-1740) → fflush. Emit cost ≈ A × draft() ≈ 4 × 0.47ms; the round graph itself is ~40-50 kernels (T=2 accept / T=1 reject; one graph, no step graphs at max_drafts=1).

**Key structural facts:**
- The 6 drafters share weights (dense_/experts_/dhead_ one pointer, mtp.cpp:715-717), separate arenas/streams — their round graphs are MUTUALLY INDEPENDENT: all A launches could overlap TODAY.
- Per-slot head = native_mmvq 2560→40,525 (~44MB head read per slot, 4-6× per window); a multi-row variant (native_mmvq_il, verify.cpp:1815) ALREADY EXISTS.
- Rollback on rejection costs ZERO (selective commit: keep-prefix staging; rejected rows write -1 cells; stale drafter K/V never read — invariant at mtp.cpp:1581-1583). Do NOT add snapshot machinery.
- Async draft APIs to model exist: chain_launch/chain_poll (mtp.cpp:1779-1859).

**R33 PLAN (top 5, by ms of the 1.87ms emit):**
1. **De-serialize drafts: launch-then-poll across slots** (~1.0-1.2ms): new `draft_launch()`/`draft_poll()` on MtpDrafter (staging+launch, no waits; modeled on chain_launch/poll); emit loop = enqueue all A residual copies + all A round graphs → poll all h_out_[0] → one sync per stream. GPU overlaps A-way; emit → max(draft_i) + launch overhead ≈ 0.6-0.9ms.
2. **One batched draft head over slots** (~0.3-0.5ms, ×#1): capture body-only round graphs (stop after mixer quantize at mtp.cpp:1223), event per drafter stream, then ONE native_mmvq_il A×2560→A×40,525 + argmax_rows(A) + row_top_prob(A) + map_ids(A) + an A-slot mtp_select variant writing each drafter's m_out_. Head read ONCE.
3. **printf/fflush out of the emit window** (0.05-0.15ms, trivial — buffer BT lines, flush after timing).
4. **Batch staging**: put()/fence/h_out_ resets + all residual copies before the first launch (fold into #1).
5. **(If proposals/slot raised)** extend #2 to A×depths head GEMM; capture_step already isolates depth.

**Not worth touching:** commit (async F-196), rollback (zero-cost by design), residual copies (80KB stream-ordered, free after #1).
**Validation:** STRATA_MTP_PROFILE=1 (head gemv/argmax/select stages), STRATA_DRAFT_TIMING=1, STRATA_VERIFY_NODES=1, bt_emit line.
