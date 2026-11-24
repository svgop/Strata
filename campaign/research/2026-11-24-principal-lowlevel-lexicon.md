# Principal low-level lexicon and formulation audit — Qwen3.8-Flash-Next on Strata (2026-11-24)

Operator brief: formulate as a principal inference engineer — better code, better logic, better code
routing. Name ≥ 60 terms that draw out that expertise, then investigate the engine *through* them.

Method. The lexicon was not written from a textbook and then applied. Each term was met first in the
tree (headers, kernels, the service loop) or in the journal's measured record, then named by its
domain term, then bound to its site (`file:line`) or its measured number. A term the engine does not
own is marked as such. The investigation (Part II) then reads every cluster back against the tree and
the journal and separates three kinds of ground: **live** (in the standing), **banked** (falsified or
retired with a mechanism — a map edge, law 8), **open** (unpriced or priced against a stale map).
Nothing open is proposed without its measured term (law 15) and its cheapest falsifier (law 4).

The cost map the lexicon is priced against (F-344/F-414, 11-20; `decode timing` lines, 11-24):

    M1 window  ≈ 13.6 ms at 3.10 committed rows  →  ≈ 228 t/s
      = GPU dense execution     ≈  8.8 ms   (65%)   "GPU-reach wait" is the GPU RUNNING, not idle
      + CPU-expert drain        ≈  2.9 ms   (21%)   ~62 µs per miss, ~1 miss/layer, linear in misses
      + window boundary         ≈  1.9 ms   (14%)   emit 0.4 + adapt wait 0.7 + commit/draft
    wall fit: ms = 8.4 + 1.57 × T   (R² = 0.99, 450 samples)  →  marginal row = 19% of the first

Everything below is read against that arithmetic.

---

## Part I — The lexicon (10 clusters, 84 terms)

### I. The speculative window — the unit of decode

1. **Verify window.** T ≤ 8 rows (the last accepted token + T−1 drafts) pass through a layer in one
   launch; the row cap and the spec depth are ONE constant (`kVerifyMaxT ≡ K_MAX`, T65 wall,
   `verify_kernels.hpp:21-29`). Live.
2. **Two-half recurrent state.** The GDN recurrence is the one state a speculative row cannot
   overwrite: the verify half runs all T tokens read-only, the commit half replays `n_keep` once
   acceptance is known — 113 MB of state, no snapshot (`verify_kernels.hpp:4-12`). Live.
3. **Bonus token.** Row 0's own argmax is emitted free whenever any draft survives; a window yields
   1 + accepted. Live.
4. **Conditional acceptance p_i.** Per-position EMA of "draft i matched, given i−1 did"
   (`dp_ok/dp_tested`, `generate.cpp:16001`). The engine's most leveraged scalar: at p = 0.76,
   dR/dp = 1/(1−p)² ≈ 17 rows per unit — +0.01 in p is +5.5% rows at fixed T.
5. **Censoring rule.** Only positions ≤ accepted+1 carry a label; a deeper position is *untested*,
   not rejected (`controller.hpp:94-96`). Live in the W1 feed.
6. **Logistic survival estimator.** σ(ω·logit q_max + β_k), one shared slope, K_MAX intercepts,
   damped IRLS every 100 windows (vLLM's shipped form; `SurvivalEstimator`). `STRATA_SPEC_LOGISTIC`, off.
7. **Cost-model controller (W1).** argmax_T E[tokens]/step_ms(T) against the measured wall
   8.4 + 1.57·rows; cut-only, never raises T (`generate.cpp:16004-16038`). Live.
8. **Row efficiency.** A depth-8 solo window runs 30–41% row-efficient; the controller trims to
   where the marginal row still pays. Live.
9. **min_p walk.** The draft chain ends at the first draft under `--spec-min-p 0.5`; this, not
   `--spec 6`, sizes most windows. Live.
10. **Teacher-forced chain.** `chain_launch` runs the whole draft chain as one async launch, its
    early steps force-fed the in-flight window's drafts (`mtp.hpp:151-165`). Live.
11. **Mid-flight draft abort.** AdaEDL entropy gate (1 − γH < λ) and a survival floor stop the
    chain at the decay knee. `STRATA_SPEC_ENTABORT`, off.
12. **Tokens/window (M3c, "H5").** 3.10 rows at M1 vs 4.88 at conc on the same boot (F-414) — the
    conc lane already shows the window shape M1 lacks. Open (F-416: +2.4% inside band at that load).

### II. The MTP drafter

13. **MTP head.** One shipped multi-token-prediction layer: cell i pairs the main model's final
    residual at i with token i+1 and predicts i+2 (`mtp.hpp:3-7`). Live.
14. **Residual tap.** The drafter reads the verifier's `window_R` — the final multi-stream residual,
    hc·n_embd per row — through `set_window_R`. Live.
15. **Draft vocabulary.** The draft head over a 40K subset (`draft_vocab_en.bin`): fewer logits
    bytes per step. Live.
16. **Q4 draft weights.** `--mtp-q4 all`: Q4_0 copies of the Q8_0 projections and head. Drafts only;
    the verify window decides every token. Live.
17. **Per-stream hidden norm.** `--mtp-hnorm stream`: each hyper-connection stream normalized alone
    (llama.cpp's qwen4exp form). Live.
18. **Dense-over-speculative attention.** The drafter's attention is dense below 2,051 cells, so
    rejected rows never touch indexer state and are simply overwritten (`mtp.hpp:16-18`). Live.
19. **All-resident draft experts.** 512 draft experts, 708 MB, never miss. Live.
20. **Drafter retraining (RT2).** The retrain lane's whole purpose is term 4: a higher per-position
    acceptance. In the workspace (`campaign/retrain/`); `windows.npy` index absent — known gap.

### III. Expert residency — the three-tier MoE

21. **Three-way split.** Every routed expert is served from VRAM (grouped kernel), PCIe staging
    (`fetch_blobs`), or the CPU pool; the hit/miss split is the whole design
    (`expert_cache.hpp:3-7`). Live.
22. **Residency table.** `(layer, expert) → slot | kNotResident`, int32 not bool — "a table that has
    to be re-derived from a bitmask is a table that will be re-derived differently" (`expert_cache.hpp:35-38`).
23. **Hit rate h.** Ten-fold leave-one-out 0.6447 vs compulsory-miss 0.4864; the standing reads
    ~96.5% of *entries* resident (diag boot). Live.
24. **Per-layer admission.** Layer l may use slots [l·q, (l+1)·q); the one global counter handed
    26 layers' worth of slots to position 0 — 2.97% hits (R4.2g). Live.
25. **Adaptive tier.** Decayed usage ranks victims (`--adapt-decay 0.85`); refill async on its own
    stream. Live.
26. **Victim protection.** PINCAP 48 (profile-rank pin), MIN_AGE 2 (admit-thrash guard), DORMANT 64
    (`generate.cpp:10637-10644`). Live.
27. **Learned profile.** `rank_learned_profile`: resident → heat → prior → index; saved every 30
    rounds. Live — and a hazard (next).
28. **Profile poisoning (F-271).** Synthetic-traffic boots accumulate a profile that then fights real
    traffic: 25,303 exchanges/boot against the 2.7–3.1k running signature; M1 down 2.2× silently.
    Every battery pins one copy for both sides.
29. **Routing-signature transient.** The first request after a signature change re-earns residency:
    2–3 PCIe expert reads against 1,045–3,788 warm, +1.5 ms/window (m1-anatomy, 11-24). Open — L1.
30. **Device-planned group.** All ten routed experts resident → `resident_plan` builds the pool's
    layout on the device; the host-plan and PCIe waits become no-ops through `*skip`
    (`verify_kernels.hpp:149-163`). Live (`STRATA_VERIFY_DEVICE_PLAN=1`).
31. **Partial plan.** Resident entries device-planned; only the CPU rows' flag is waited
    (`kPlanPartialSkip = 0x80000000`). Live.
32. **PCIe-staged blobs.** `fetch_blobs`: coalesced 16-byte loads from mapped host memory into VRAM
    ahead of the grouped kernel. Live.
33. **pcie_frac.** The share of a layer's misses routed to PCIe staging rather than the CPU; standing
    0.35, `--pcie-mode dma`. Live.
34. **VMM-segmented arena.** One reserved range backed by physical segments: VRAM shrinks and grows
    while every captured-graph pointer holds (`expert_cache.hpp:117-146`). Live (elastic KV).
35. **Slot guard.** Bytes kept past each sized slot because the prompt MMQ reads past a down matrix.
36. **Complement pin.** The non-resident experts' arena pinned against the page cache; thrashes
    against NVMe when the learned profile diverges (F-417). Live, F-441–443 refactored.

### IV. The CPU expert pool

37. **Split GEMV phases.** Up-projection (phase A) and down-projection (phase B) each cut into ~3
    tasks per thread so the tail is short — two barriers per layer with a miss (`pool.cpp:1042-1053`). Live.
38. **Epoch-carrying claim.** `head_` packs (epoch, n, i); a straggler's CAS fails on the epoch, so
    correctness never needed the repark (issue #29; `pool.cpp:873-896`). Live.
39. **Activation quant.** `act_quant_q8_1`: per-32-block scale, ISA-dispatched (AVX-512/2/1). In
    split mode it runs SERIALLY on the host thread between the phases (`pool.cpp:1050`). Priced out:
    ~0.5 µs × 48 ≈ 24 µs/window — a fold into phase A's workers is exact (each task owns a contiguous
    ff slice) but buys nothing at ~1 miss/layer.
40. **Park / spin / sleep.** `_mm_pause` spin; condvar sleep after `STRATA_POOL_SPIN_US` (standing 100 µs).
    **F-346: 100 → 2000 moved nothing — the worker park is not the carrier.** Banked.
41. **Per-miss cost.** ~62 µs per miss, linear: 0.12 entries/layer → 0.36 ms/window, 1.31 → 3.93 ms
    (decode-timing, 11-24). 2.2–2.5× the 35 µs DRAM floor for 1.38 MB at ~40 GB/s.
42. **Pool diet.** Trailing repark waits skipped (`STRATA_POOL_DIET=1`, +4.06% M1, F-227). Live.
43. **Per-layer granularity.** `run_split` is called ~48× per window at ~1 miss each: the F-417 note
    read this as "machinery dominates the GEMV". **The measurement says otherwise** — host plan +
    actq + jobs ≈ 0.13 ms/window of the 2.9 ms; the term is the pool's drain itself (`CPU`).
44. **VNNI dot product.** `s2_expert_vnni_q`: AVX-512 VNNI int8 activations × q8 weights; 22,257 zmm
    instructions, no silent AVX2 fallback (F-348 audit). Live.
45. **X3D V-cache.** 2 × 64 MiB L3 on a 9950X3D; a missed expert is DRAM-cold by definition (the
    misses ARE the cold tail), so the L3 buys the pool little.
46. **CCD isolation.** Engine pinned to the isolated CCD (workers 9–15, host 8, aux 24); same-CCD
    contention from housekeeping load measured 232.6 vs 120.9 t/s. Live.
47. **SMT sibling contention.** Maya's calibration: fewer threads beat more on memory-bound matvecs
    (15 → 41 t/s); "our pool is 15/15; a 12/10 retest is cheap" — **never run.** Open, cheap.

### V. Handshakes — the GPU↔host protocol

48. **Mapped-memory flag.** The GPU spins on a host-written word: `while (*flag < value)
    strata_spin_pause()` (`verify_kernels.cu:1311-1313`). Live.
49. **System-scope fence.** `__threadfence_system()` after the spin and before the publish: the
    store must leave L2 for host memory before any thread proceeds.
50. **Three waits per layer.** flagA (host plan), flagB (PCIe staging), flag (CPU rows). F-414 on the
    standing: A 4.8–5.2 µs (dead), B 9.8–24 µs (covered), **CPU 37.9–59.1 µs — exposed, ~1.35 ms/window.**
51. **Fused wait+copy.** `wait_copy_i32_from_mapped`: thread 0 spins, the block then copies — one
    node fewer per layer (~2.4 µs on gfx1201). Live.
52. **Doorbell.** The host's publish of a layer's plan; `router+plan+bell` 0.67 ms/window in the
    F-218 stage profile.
53. **Bell thread.** The serve thread tight-spins during decode (`if (busy) continue;`); its 2 ms
    condvar is the idle path only (F-347).
54. **Handoff publish.** The last block to finish raises the flag after a system fence
    (`handoff_publish`) — the split-overlap lanes' GPU-to-GPU handoff.
55. **Speculative publish.** Predict layer k+1's miss rows and compute them under layer k's GPU
    time. **FALSIFIED by routing data: adjacent-token same-layer Jaccard 0.232 / median 0.176,
    subset 0.3%** (F-348). Banked: "real work that must be made cheaper or hidden by structure,
    not predicted away." The 11-20 lookahead note re-opens it without citing this — it does not.
56. **Cohorts.** Two verifiers so B's pool work runs under A's GPU window (`STRATA_COHORTS=2`,
    +24.8% conc, F-233). Live — and the reason conc reaches wait 0.51 ms where solo eats 8.79.

### VI. GPU execution structure

57. **Per-T captured graphs.** `round_exec_[9]`, `step_exec_[9]`, `prefill_exec_[9]`: one graph per
    window width; the T65 wall ties every array to the row cap.
58. **Conditional-node tail.** `win_cond_tail` drops the WHILE handle so the body runs exactly once
    per launch, re-armed by `cudaGraphCondAssignDefault` with no host call (CUDA 12.4+).
59. **Node-count verdict.** −344 nodes/window ≈ −0.17..−0.33 ms across four one-boot arms: "node
    counts are not the binding constraint at our spacing." **Banked negative** — the megakernel
    re-frames as boundary elimination, not node deletion.
60. **PDL.** Programmatic dependent launch; the early-launch stretch keeps a profiled chain intact.
    A Windows −40% lived here — the sustained-reasoning probe is mandatory before any launch-class arm.
61. **Stream priority.** The drafter's stream highest; F-206's `draft_launch/draft_finish` enqueues
    every step back-to-back so slot t+1's GPU work no longer waits for slot t's host sync.
62. **Spin primitive.** `strata_spin_pause` (`dp4a.hpp:47`): each poll of a mapped word is a PCIe
    read; the poll period sets how soon the GPU sees a 50 ns host store.
63. **Stage profile.** `gpu_stamp` writes `%globaltimer` at every stage boundary;
    `STRATA_VERIFY_PROFILE=1` prints the per-stage ms/window. The instrument Part II runs today.

### VII. The dense chain — 65% of the window

64. **Fixed window cost.** 8.4 ms at T=1 = 48 layers × ~175 µs of dense execution. The "GPU-reach
    wait" (≈ 8.0 ms at T=1) is the GPU RUNNING between handshakes — not idle. The idle is the ~62 µs
    per miss the GPU spends at the CPU-rows flag.
65. **Hyper-connections.** Four residual streams; A_m (read mix, `hc down`), B (write broadcast) and
    A_r (cross-mix, `hc up`); two sites per layer (hc0 before attention, hc1 before the MLP).
66. **hc share.** F-218 stage profile: hc0/hc1 down + up + norm = **3.12 ms/window, 25% of 12.32**
    — the largest single kernel class after the MoE tail. Priced against a stale map; Part II re-reads it.
67. **hc absorption (R37).** Fold A_mᵀ into the adjacent weight: W_in·(A_mᵀH) = (W_in·A_mᵀ)·H — the
    static part exact, the dynamic part a rank-s correction. **Never executed, never priced against
    the standing.** Open.
68. **GDN block.** conv(3) → SiLU → L2 → α/β → recurrence → out-proj. The one-pass identity (R28):
    readout from the PRE-update state, halving the serial chain; 1 read + 1 write is the floor at
    m = 1 (the WY negative). Repriced "small class" post-T1.1 — correct against today's arithmetic:
    the GDN step is a slice of the 1.57 ms marginal row.
69. **Deferred log-space decay.** λ_t = λ_{t−1} + ln α_t: removes the per-step exp SFU from the
    head of the serial chain; fold at λ < −30. Same class as R28.
70. **QSA sparse attention.** Indexer top-k over cells; attention is the context-scaling stage
    (0.42 → 4.09 ms/window across the profile legs).
71. **Sign-bit ordinal map.** `(b & 0x80000000) ? ~b : (b | 0x80000000)`: IEEE total order for
    radix select; branching on `x ≥ 0` would wrongly capture −0.0. **Shipped** (`qsa_select.cu:31`).
72. **Head cost.** 521.5 MB Q6_K output head, 0.43–0.78 ms/window; the requant rider is MMLU-gated.
73. **Shared expert.** Always-resident dense expert beside the routed ten; `shared+quant`
    0.63–0.85 ms/window; its own side stream in the drafter.
74. **Arithmetic intensity at T = 4.** Each dense weight byte is used four times; the chain is
    HBM-bound, not FLOP-bound — 1400 GB/s measured on the expert math against the head's 1.18 TB/s.

### VIII. Memory hierarchy

75. **Pinned ring.** Four slots, event-guarded; pageable sources bounce through it because the
    runtime's staging is not ordered the way a pinned copy is (#1996, a 31-minute fault class).
76. **Pinned-alloc lock.** A `cudaHostAlloc` takes the driver's lock and can wait on the device;
    rings are grown before a read that runs beside windows.
77. **Mapped host memory.** The GPU reads pool rows straight from worker-written host memory
    (volatile reads — on RDNA a plain store can sit in L2).
78. **Huge pages.** mmap + THP on the expert tier; a 31.64 GiB shadow-store mapping.
79. **First-touch faults.** Leg 1 reads 184.8 against 205–226 warm (F-360): the discarded leg exists
    for this, and every new mapping owes the prewarm its coverage.
80. **Page-cache re-warm.** The server's `prewarm` lifecycle re-touches every store the engine reads
    before the first request (today: 21 s replay, 98.6–99.6% hit).

### IX. Verification, sampling, numerics

81. **Exact-match verification.** A greedy draft survives iff identical; M3b is the canary under
    this contract (law 10).
82. **Coupled draft sampling.** Sampled requests draft with the target's chain and the target's
    Philox draw (`STRATA_SPEC_COUPLED`); **probability-matched acceptance** (`spec_prob.hpp`) makes
    the sampled form lossless. **Gumbel-max coupling**: argmax(log p + Gumbel) ≡ an exact sample of p
    (`sampler.cu:66-72`). **Typical acceptance** (τ = 0.2) is sampled-traffic only — output-changing.
83. **Multi-block argmax.** The one-block argmax is slow on sm_80–89; `argmax_rows` splits a row
    across ≤128 blocks with a last-block merge, bitwise the greedy pick.
84. **FMA shape preservation.** `__fadd_rn(0.0f, row)` and one fmaf per expert in order: parity
    proofs here are bitwise, never epsilon (`verify_kernels.hpp:195-202`).

### X. Code routing and structure

85. **Residency-dependent dispatch tree.** Per layer: all-resident → device plan; partial → device
    plan + CPU wait; else host plan → three waits. Decided from the router's output after it lands.
86. **Skip-word gating.** A captured wait becomes a no-op at replay when `*skip == value`: the graph's
    own branchless dispatch, no re-capture per shape.
87. **Virtual expert source.** The `ExpertSource` vtable (blob / prefetch / begin_layer…), ~480 calls
    per window. Priced: ~µs — negligible, and the owner law wants it.
88. **Env-gated knob.** Every arm is a `STRATA_*`, default off, one-line rollback in the launcher;
    the default path stays byte-identical.
89. **Compile-time invariants.** `static_assert` ties K_MAX, kVerifyMaxT and every graph array (T65);
    the window-width wall's nine asserts turned three silent corruption classes into build failures.
90. **Parity harness.** `dm_plan_parity`, `cvec_parity`, `gdn_rec_parity`: a bitwise proof precedes
    any arm — law 11's "instrument the instrument".
91. **Owner law.** `GgufFile`, `battery-lib.sh`, the pack-manifest loader: a second implementation is
    the defect, not the shortcut.
92. **Warm protocol.** Prewarm → one discarded leg → legs 2..N; means only (laws 1, 11).
93. **Interleave.** ABAB / 3v3 inside one window; the box drifts ±8% across hours — a single-boot
    number is a rumor.
94. **Adopt band.** ≥ +4% on the target and no lane below −4%; inside the band is not a win.
95. **Scoping before verdict.** A lane-negative result is a scoping problem (law 4b): the next move is
    to apply the arm only where it pays, before it is closed.
96. **Pathway order.** Remove a stage, fold two into one, move work under an existing wait — THEN
    add a mechanism (law 9). The order is the whole doctrine.

(96 terms; the brief asked for 60.)

---

## Part II — Investigation (written after the diagnostic leg; see below)

### II.0 What the diagnostic leg established (today's map, standing recipe, one boot)

Boot pid 669135, recipe `config/custom-iq3_s-8081.json` verified on the engine process
(`--expert-cache 9900 --spec 6 --mtp-q4 all --mtp-hnorm stream --pcie-frac 0.35 --pcie-mode dma
--adapt-decay 0.85`), cvec banner `armed add:8-8+project:1-47`, the model's own jinja, prewarm
`done` (21 s replay, 98.6–99.6% hit). `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` confirmed
in `/proc/<pid>/environ`. Profile pinned (`2ef052c1…`), restored after, md5-verified. Leg 1
discarded (197.1 — F-360's class); leg 2 measured: **M1 206.3 t/s instrumented** (the `gpu_stamp`
nodes cost a few percent; a diagnostic, not an arm, no tracker point). Load 2.53. Evidence:
`campaign/results/lex-2026-11-24/{probe-leg2-profile.txt,engine-anatomy-lines.txt}`.

**Disclosed confound:** the kernel's isolated set was empty (`/sys/devices/system/cpu/isolated`,
no `isolcpus` on the cmdline; up since 10:57), so by the launcher's own law this boot ran
**unpinned** — neutral on a quiet box, but every absolute millisecond below is an unpinned-boot
number. The stage shares are robust to it; see II.5.

**The M1 request (558 windows, 3.20 rows, 2.67 tokens/window), GPU stages, ms/window:**

| stage class | GDN layers | QSA layers | total | share |
|---|---|---|---|---|
| VRAM hits (grouped expert kernel) | 1.75 | 0.55 | **2.30** | 17.6% |
| copy+combine (holds the CPU-rows spin) | 0.77 | 0.26 | **1.03** | 7.9% |
| router+plan+bell | 0.71 | 0.23 | **0.94** | 7.2% |
| shared expert + act quant | 0.68 | 0.23 | 0.91 | 7.0% |
| waitA / waitB / PCIe grp / gap | 0.32/0.17/0.11/0.14 | 0.10/0.06/0.03/0.04 | 0.97 | 7.4% |
| attention block (qkv, conv, ab, z, rec / k,v,q, attention, out-proj) | 2.30 | 2.09 | 4.39 | 33.5% |
| hyper-connections: down / up / norm | 0.67/0.86/0.15 | 0.23/0.27/0.02 | **2.20** (1.00/1.13/0.07) | 16.8% |
| head | 0.36 | — | 0.36 | 2.7% |
| **total** | | | **13.09** | |

Wall 14.70 ms/window (batch summary) = run 13.41 (GPU-reach 10.18 + CPU experts 2.75) + commit
0.01 + emit 0.54 + adapt wait 0.73. Host split (decode timing, per window): GPU-reach 8.96 +
per-layer host 2.19 **[plan 0.09, actq 0.05, jobs 0.13, CPU 1.74]** + stage 0.07 + commit 0.01 +
**draft 1.06** + adapt wait 0.24 / join 0.03; 0.72 entries/layer on the CPU, 6.9 swaps/window.
The conc request on the same boot (≈700 windows, 5.0 rows): 13.8–14.2 ms = reach **0.36–0.64** +
CPU 2.5–3.3 + commit 0.30 + emit 0.31 + adapt 0.68.

### II.1 Three corrections to the working model (law 8: on the record)

1. **"GPU-reach wait" is execution, not starvation.** ≈ 8.0 ms at T=1 and 8.96 at T=3.2 is the GPU
   running the dense chain between handshakes. The GPU's actual idle inside a window is the
   CPU-rows spin (F-414: ~1.35 ms exposed) — `copy+combine` 1.03 + waits 0.65 is its container —
   plus the window boundary. F-414's "solo starves the GPU" is right about the boundary and wrong
   about the reach term; the equation in II.2 replaces it.
2. **"Machinery dominates the GEMV" is false on measurement.** The F-417 note (11-20) read
   `run_split`'s two phase setups and the serial quant loop and priced the pool at "machinery
   dominates at ~1 miss/layer". Today's host split: plan + actq + jobs = **0.27 ms** of the 2.75 ms
   CPU term; the other 1.74 (decode timing) / 2.75 (summary) is the pool's **drain**. The serial
   `act_quant_q8_1` is ~0.5 µs × 48 ≈ 24 µs/window. A fold into phase A's workers is exact (each
   task owns a contiguous `ff` slice, `pool.cpp:955`) and buys nothing. Banked: priced out.
3. **The boundary has a third term.** F-414 named emit (0.4) and adapt (0.7). The draft is
   **1.06 ms/window serial** in solo — the loop calls the launch→spin→sync `mtp.draft()`
   (`generate.cpp:17160`), the form F-206 split for the batch lane and never for solo.

### II.2 The window as one equation, and its derivatives

    t/s = 1000 · R / W,     W = F + m·T + B

* **F = 8.4 ms** — the T=1 dense chain (attention blocks, hc, head, shared expert, VRAM hits at
  U(1) ≈ 10 experts/layer). HBM-bound: the VRAM-hits stage implies ~1 TB/s (≈ 34 distinct experts ×
  48 layers × 1.38 MB in 2.30 ms) on a 5090 — **at its floor at this quant**.
* **m = 1.57 ms/row** — ≈ 0.26 of extra dense execution + ≈ 0.6 of extra CPU misses
  (U(4)/U(1) = 3.4× distinct experts) + ≈ 0.5 of commit/draft/emit growth + the deeper attention.
* **B ≈ 2.3 ms (solo)** — emit 0.54 + adapt wait 0.73 + draft 1.06 (commit 0.01). Cohorts cut B to
  ≈ 1.3 by carrying other slots' rows under it.
* **R = 3.21**, T = 3.20.

The lever classes are the partials:

* ∂/∂R — **acceptance** (drafter quality): at p = 0.61, dR/dp = 1/(1−p)² ≈ 6.6 rows/unit →
  +0.01 p ≈ +3.3% R. The retrain lane's whole purpose; blocked on `windows.npy`.
* ∂/∂T at fixed R — cut-only controller already does this (W1).
* ∂/∂B — **the boundary levers (II.3 L1–L3)**: reorderings of independent work. Bitwise output.
* ∂/∂m — deeper windows pay better once B and the per-miss cost fall (composition, law 4b).
* ∂/∂F — byte-count levers (requant: MMLU-gated, a named capability axis) or fold levers (R37).
* The **hit rate** moves m's CPU part: F-344's 0.6–0.9 ms per point stands; the VRAM is full
  (917 MiB free against the vision reserve) so slots cannot buy it, and F-412 falsified the
  profile-calibration delivery (+2.1pp offline → +1.5% live).

### II.3 The priced lever queue (needles priced before they run — law 15)

| # | lever | term | needle (ms/window → M1) | cheapest falsifier (law 4) | status / overlap |
|---|---|---|---|---|---|
| **L1** | **Draft under the emit** — `draft_launch` → emit → `draft_finish` in the solo loop | B | ≤ 0.54 → **+3.7%** | one code read: are emit and the chain data-independent? (yes: `fflush` at :17141 precedes `draft()` at :17160 and shares nothing with it) | **new**; F-206's split already exists for the batch lane — a MOVE (law 9 §3) |
| **L2** | **Adapt wait as a stream dependency** — the window graph waits `adapt_ev` on-stream; the host never `cudaEventSynchronize`s | B | 0.3–0.73 → **+2–5%** | read `apply_pending(true)` (:16842, :10760): is the sync guarding a host-side table write the device could order? | **new**; ADAPT_OVERLAP took 1.03 → 0.28–0.62 (F-333 era); this is the remainder |
| **L3** | **Speculative next window on one GPU** — launch K+1 on the guess K is accepted whole; K+1's rows fill K's stalls | B + in-window idle | up to 2.3 when right (p² ≈ 37% at T=3) → **+6–8% net** | the pipeline-windows-2 substrate exists (`generate.cpp:16074+`); falsifier = its single-GPU construction cost + the snapshot-restore tax | **structural** (F-223→F-233's solo form); law 6 |
| **L4** | **L1 prefix-memoized residency** (m1-regression INVESTIGATION) | m (CPU part) on first-of-signature traffic | +8–10% on that class | anatomy control (same-burst removes it) — done | open, priced 11-24 |
| **L5** | **Deeper windows with B cut** (H5 / M3c) | R·T | F-416: +2.4% inside band at the old B | re-run the named ladder AFTER L1–L3 land | open; composes — the controller cuts less as m and B fall |
| **L6** | **hc absorption (R37)** — fold A_mᵀ into the adjacent weight | F | down-mix 1.00 → net 0.4–0.7 → **+3–5%** | one kernel read: is THIS checkpoint's A_m static or DHC-dynamic? then a parity harness | never executed; static-only is lossy at n=4 → **MMLU gate** |
| **L7** | **Router+plan+bell 0.94** — R29's surviving term: the device writes the plan to mapped memory; a pool-side dispatcher consumes it | F (handshake) | ≤ 0.5 → +3–4% | stamp the bell segment alone (`STRATA_DM_WAIT_STATS` already splits A/B/CPU; the bell is inside `router+plan+bell`) | R29 has no outcome entry; the device plan (T1.1) and dm fold are its descendants |
| **L8** | **IQ2 expert requant** (VRAM hits 2.30 → ~1.6, misses −30%, more slots per GiB) | F, m | +7–10% ceiling | the pack exists (`q2_0-rebuilt`, INCOMPLETE) | **capability axis** — law 16: named, MMLU ×2, Pareto rule (F-450) |
| **L9** | **Acceptance (retrain, RT2)** | R | +0.01 p → +3.3% | rebuild `windows.npy` (known gap) | the lane exists; blocked on the index |
| L10 | Head requant (0.36) | F | +1–2% | — | MMLU-gated; small |
| L11 | Pool thread count (Maya 12/10) | CPU drain | ±? | `--pool-workers N` config ladder | **re-opened by II.5**: on an unisolated box the pool sits on housekeeping cores again |

**Not levers (banked this session or already):** shared+quant hoist (F-362: G=1, already once per
layer — the 0.91 ms is the shared expert's real cost); the serial quant fold (24 µs/window); pool
spin threshold (F-346 null); speculative publish / RouterLookahead compute (F-348: Jaccard 0.232 —
the 11-20 note re-proposes it without citing the falsification; it does not re-open); node count /
conditional nodes (−344 nodes neutral); slot count (VRAM full); profile-calibrated hit rate (F-412).

**Ship order (law 6: structure first, then compose):** L1 + L2 are one commit — the solo boundary
re-routed, bitwise output, no gate owed beyond M3b-in-band — expected **+5–8% composed**, which
clears the band where either alone might not. Then L3 (the structural one). L6/L7 in parallel as
kernel-side projects. L8 only as a labeled shelf variant. L5 re-laddered after L1–L3.

### II.4 The code-routing reading (what "better code routing" means in this engine)

The per-layer residency dispatch is already the right shape: the graph is captured once and a
`*skip` word makes each wait a no-op at replay (`verify_kernels.hpp:149-175`) — branchless, no
re-capture. The dispatch that is NOT yet intelligent is at the window boundary, where three
independent streams of work — host emit, drafter GPU chain, adapt PCIe copies — are routed through
one host thread in series. The pool's per-layer dispatch (48 `run_split` calls/window at ~1 miss)
is forced by the data dependency (layer l's combine needs layer l's rows) and is not a routing
defect; its cost is the drain (II.1 §2). The remaining true routing defects are therefore L1, L2,
and R29's bell (L7): host→GPU→host round trips that exist because of the pre-device-plan shape,
not because anything waits on them.

### II.5 Environmental finding: the isolated set is empty

`launch::isolated_cpus()` reads the kernel's isolated set; it is empty, and the cmdline carries no
`isolcpus`/`nohz_full`/`rcu_nocbs`. By the launcher's law the engine ran **unpinned all day** —
the 14:08 bench (236.9), the anatomy runs, this diagnostic. On a quiet box that is measured
neutral; under housekeeping load the 11-18 falsifier measured same-CCD 232.6 vs other-CCD 120.9.
Consequences: (a) today's numbers and the 11-20 bank may differ on the pin axis, not the hour —
the incline comparison should name it; (b) the pool's placement (one worker per isolated core) is
not in force, which re-opens Maya's thread-count question (L11) for as long as the box is
unisolated. Restoring isolation is the operator's boot configuration; the engine-side check is one
interleave, pinned (`STRATA_ENGINE_CPUS=8-15,24`) vs unpinned, same load.

### II.6 What the lexicon proved about the expertise it names

Of 96 terms, roughly two-thirds are machinery the tree **owns and runs** at its principal-grade
form; about a dozen are **banked edges** with mechanisms (F-346, F-348, F-362, F-412, the node
count, the serial quant, R28's repricing, the slot count, WY at m=1, …); **eight are open and
priced above** (L1–L9); ten are doctrine (the laws that govern the work). The lexicon's value is
exactly that ratio: it shows where the engine already is at the form the expertise would build,
and it makes the remaining ground enumerable — three boundary re-routings, one structural window,
two kernel folds, one quantization axis, one drafter lane.
