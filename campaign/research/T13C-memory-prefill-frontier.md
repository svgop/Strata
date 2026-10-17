# T13C — MEMORY SYSTEM + PREFILL PATH: line-level audit and external frontier (2026-10-16)

Scope (operator directive): (A) code audit of expert-cache eviction/residency, foresight/swap,
PCIe staging, KV layout/growth/streaming, and the prefill path (GDN chunked kernel, QSA kernel,
dense projection MMQ — where is the a1444e33-style dense path missing?); (B) external dig
BEYOND T9/T10/T11/T12 (all known items marked **known**). READ-ONLY audit; this file is the only
write. Server untouched. Calibration: standing recipe = launch-custom-iq3s-8081.sh env +
custom-iq3_s-8081.json; targets single-stream 250 / conc 300; prof221/F-218 cost map assumed.

---

## PART 1 — CODE FINDINGS (all file:line read this session, tree @ F-224 state)

### 1.1 The residency "policy" is really an ADMISSION policy — there is no eviction anywhere

- `ExpertCache` is fill-once. `admit()` (src/core/expert_cache.cpp:638-656) returns a slot only if
  a free one exists (`next_free_` global counter; `per_layer_` quotas exist but default OFF,
  include/strata/core/expert_cache.hpp:237 — "per_layer_ off (the default) leaves next_free_ as
  the only admission counter"). Comment at :653: "full: no eviction, deliberately". Once the
  9,337 slots fill (profile-ranked sequential fill, data/expert-profile.bin), the ONLY mechanism
  that changes residency is the adaptive tier.
- The adaptive tier IS the eviction policy: every `adapt_every` rounds it swaps up to
  `adapt_swaps`=96 of the most-routed missing experts into the VRAM tier, with decayed usage
  (generate.cpp:677-709, `--adapt-decay 0.85` standing), prediction credits
  (RouterLookahead::take_credits, expert_source.hpp:281; STRATA_PREDICT_CREDIT=1.0,
  STRATA_PREDICT_BUDGET=2 standing) and victim protection (STRATA_ADAPT_PINCAP=48 /
  MIN_AGE=2 / DORMANT=64 standing). Exchange machinery: FileExpertSource::stage_exchange /
  commit_exchanges (src/core/expert_source.cpp:3345-3428), fs_hold_staged (:3373).
- Consequence: every "better eviction policy" candidate (LRFU #2458, tqllm decayed-frequency,
  REAP placement — all **known**) is here an *admission-reordering* candidate competing for the
  same 9,337 slots. The +2.1 slots/GiB slope of T12-MEM stands; the marginal lever is WHERE the
  slots go, not how they leave.
- kvg_ensure (generate.cpp:6881-6983) grows the elastic K/V by unmapping expert-cache chunks,
  with a heat-shuffle (hotter loan-region expert moved into a colder slot by DtoD copy, :6927-6940)
  and a cudaDeviceSynchronize before unmapping (:6948). Dormant at steady context. Elastic-EXT
  K/V port (78652d23+e6ed2a50, +165 slots) is **known** (T10 #4).

### 1.2 Foresight/swap tier: fully audited — and it is DEAD CODE under the standing recipe

- `ForesightSwap` (src/core/expert_source.cpp:5014-5277): per-(layer,expert) slots in extra VRAM;
  LRU-ish victim = empty slot else least-recently-used outside a `depth`=4 window (:5205-5217);
  admission after `admit` misses (STRATA_FS_ADMIT=1), `budget`=32 copies/card/window (:5263-5265);
  copies on one filler thread from the pinned arena (:5092-5116); `take()` consulted per miss in
  the plan loop (:3896), `note_miss()` on every miss (:3899) — a mutex acquisition per miss per
  layer on the service thread.
- `foresight_swap_from_env` REFUSES to start when STRATA_VERIFY_DEVICE_PLAN is set
  (:5256-5260: "the device would plan resident layers without the host"). Standing runs
  STRATA_VERIFY_DEVICE_PLAN=1 (launch-custom-iq3s-8081.sh:15) ⇒ **ForesightSwap is off, and
  structurally so** (**known** as a note in T12-MEM #7; the audit adds: this is a *contradiction
  inside our own queue* — the banked T8.A#3 "prefetch across the MTP draft horizon" and the
  DuoServe-class next-layer prefetch have NO host path that runs in the standing config; any
  such work must first reconcile DEVICE_PLAN with a prefetch tier, or attach prefetch to the
  staging/adapt machinery instead).

### 1.3 PCIe staging path (decode windows)

- Plan: `expert_pool_dispatch_multi` (expert_source.cpp:3694-4095). The T12-HOST #1 fix is
  **LANDED**: single-pass seen-mask plan with stamp arrays, STRATA_PLAN_ONEPASS default-on and
  STRATA_PLAN_DIFF independent-scan auditor (:3739-3827). The SVG-CUSTOM-TODO T12-HOST ledger
  still lists this as open — stale.
- PCIe share: `m = (nmiss * pcie_num) >> 8` (:3867; pcie_num/256 = --pcie-frac 0.35 standing),
  last-in-routing-order misses, `fetches < P.staging_cap && fetches < 64` (:3900); blob must be
  pinned (`src->pinned()` :3901) — the pin comes from the cudaHostAlloc/cudaHostRegister arena +
  compact complement (expert_source.hpp:576-584 pin_cache_complement; ArenaExpertSource
  device_alias :4972). `GpuPlanSink.pcie_mode`: 0 = copy-engine DMA into staging, 1 = grouped
  kernel reads the mapped arena directly, 2 = copy kernel inside the graph
  (expert_source.hpp:337-340; verify.cpp:1700-1706).
- `Verifier::fetch_dma` (src/core/verify.cpp:2604-2630): mode 0 = per-blob cudaMemcpyAsync loop;
  STRATA_DMA_BATCH=1 (standing) = one cudaMemcpyBatchAsync via `copy_expert_blobs` + a
  cudaLaunchHostFunc that raises flag B when the copies land (:2619-2629). Staging = device arena,
  `kStagingBlobs = 32` (verify.hpp:552-554, "savage-custom: 32 (was 16)... cap saturated at the
  stock value" — the cap was already doubled once; at --pcie-frac 0.35 with ~30 misses/layer the
  DMA pick wants ~16+; a 32→48 sweep costs ~18.5 MB VRAM).
- Still open from T12-HOST: the O(m²) `std::find` miss list (expert_source.cpp:3997-4002) — m is
  1-3 at decode (negligible there) but the same list runs on prefill-sized windows if the CPU-pool
  path is ever taken with more misses.
- T12-GPU item 4 ANSWERED: hits-in-combine is armed on CUDA only by STRATA_FUSE_HITS=1
  (verify.cpp:284-303: `return env == 1;` — "CUDA never had the arm"), plus conditions
  `n>1, n<=8, k==10, dec_batch, !prof_on_, !sg_gated_[grp], native_moe_combine_enabled()`
  (verify.cpp:1731-1735; native_moe.cu:186-188). Un-armed in standing. The ZADD ≡ zeroed-row+add
  bitwise argument is in native_moe.cu:46.
- Also found landed-but-unarmed: `STRATA_FUSE_WAIT` on CUDA (verify.cpp:262-269, F-205,
  "~144 nodes/window", "default off until the A/B lands").

### 1.4 CPU pool (the miss path)

- Barrier diet is **LANDED but OFF**: `STRATA_POOL_DIET` (src/kernels/cpu/pool.cpp:480-481,
  755-759, 901-904; pool.hpp:239-242) skips the trailing `wait_parked` in run_phase/run. The
  epoch-carrying claim already fences stragglers (issue-#29 lesson, pool.cpp:755-758). Not in the
  standing env → T12-HOST #2 is built-but-never-armed. Same for the "keep workers hot" idea —
  STRATA_POOL_SPIN_US=100 standing parks after 100us (pool.cpp:478); UMWAIT exists MSVC-only
  (pool.cpp:24-43), Linux spins (R4 #8 **known**).
- Pool matvec kernels: s2_expert_gu_rows_multi / down_rows_multi (pool.cpp:735-743); split_rows
  default on (expert_source.hpp:417-418). R4's parallelism-limited verdict stands.

### 1.5 KV layout / growth / streaming

- QSA KV streaming (include/strata/kernels/kv_stream.hpp:1-121): authoritative K/V in pinned,
  device-mapped host memory, identity layout; VRAM = page pool (1 page = 4 cells = one indexer
  block, 4,224 B int8) + residency map; `kv_stream_resolve` runs INSIDE the captured graphs:
  claim misses → CLOCK second-chance victim sweep → zero-copy PCIe copy kernel. Writers always
  write the host copy, slots only when resident — nothing to invalidate. This is exactly why
  decode is flat vs depth and why KV past the 32k resident window is near-free (context facts
  confirmed by mechanism).
- STRATA_KV_PREFETCH=1 (standing): next streamed layer's prefix staged on its own stream
  (src/prefill/prefill.cpp:2099-2117, kv_stage_from_host :2323/:2736; drain guard :2114-2117).
- --kv int8, --kv-resident 32768 standing. STRATA_L2_SETASIDE_MIB=60 already pins the recycled
  set over the GDN state (T9.A#4 **known**, armed).
- No new defect found in the KV path. The 262k+ enablers (QSA top-k past 262,144 cells;
  elastic-EXT K/V) remain the **known** T10 #4/#5 ports.

### 1.6 Prefill path

- **GDN chunked kernel** (src/prefill/kernels.cu:714-1063): WY-form DeltaNet in FP32 scalar FMAs;
  prep = one block per (32-token chunk, key head), T=(I+A)^-1 by in-register forward substitution
  (:810-817); scan = one block per (key head, 16 value columns), state slice in registers [4][4],
  cp.async double-buffered k/q staging, L2 prefetch of chunk+1 (:828, :912-920), 93.7 KB smem,
  128 blocks/one wave, kGdnChunkedMin=128 (:729), 25.6 MB static scratch (:1003-1015).
  STRATA_GDN_CHUNKED=1 adopted (F-224: recurrence 1.68x, prefill −4.2%) — **known**. Remaining
  headroom on this kernel = tensor-core forms: llama.cpp #29353/#30087 BF16-MMA (+16-18% pp2048,
  K==1 only — **known** T3.5) and eddoursel 50c83e0f m16n8k16 chunk-walk (recurrence 2.1x, ~4e-4
  non-bitwise — **known** T10 #3). Gated DeltaNet-2's erase/write decoupling (arXiv 2605.22791)
  is model-side — NOT portable to frozen weights.
- **QSA prompt attention**: MMA kernel `qsa_prompt_attn_batch` exists (src/kernels/cuda/
  qsa_prompt_attn.cu:1455; old-kernel fallback at prefill.cpp:2918-2922). Upstream #2756's
  STRATA_PF_ATTN_BATCH 32→128 QSA prefill-attention batch is **absent from our tree** (grep: 0
  hits) — **known** T8.C candidate, now code-verified un-ported (~9% of QSA prompt-attn time,
  bitwise via FP32-hash parity upstream).
- **Dense projections — the a1444e33 gap, precisely located AND re-priced**:
  - `Gemm::native_mmq` exists at src/prefill/gemm.cu:813-876 but sits inside
    `#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)` (:812), dispatch at :936-938;
    CMakeLists.txt:72 makes STRATA_PREFILL_MMQ HIP-only (`FATAL_ERROR ... requires
    STRATA_ENABLE_HIP=ON`, :88-89). On CUDA the dense line runs dequant_f16-scratch +
    cublasGemmEx (gemm.cu:936-951, :721) — T12-MEM #4 confirmed; dense ≈ 25-40% of chunk.
  - The K%256 hazard is handled by REFUSAL today (gemm.cu:822-828: non-256-multiple K, e.g. the
    640-value shared-expert down, stays on the dequant path) — upstream a1444e33 zeroed 4 KiB
    tails instead; the port keeps our refusal semantics or adopts the tail-zero.
  - **NEW from the artifact (GGUF header parsed this session)**: the dense tensors of the standing
    pack are predominantly **Q6_K (type 14)** — attn_qkv, attn_output (11 of 12 QSA layers;
    blk.3 is Q4_K), ssm_out, output.weight — with the router ffn_gate_inp / ffn_gate_inp_shexp /
    hc_ffn_down at **IQ1_M (type 30)** and token_embd at IQ2_S. Our strata_kq MMQ library ALREADY
    builds Q4_K/Q5_K/Q5_1/**Q6_K** instances on CUDA (src/prefill/mmq_kq_lib.cu:53-56;
    CMakeLists.txt:79 STRATA_MMQ_KQUANTS=ON default). ⇒ The a1444e33 port is SMALLER than the
    "~130 lines + 5 sites" estimate suggests: lift the HIP gate, wire `STRATA_DENSE_MMQ` on CUDA,
    and the biggest GEMMs (Q6_K) consume kernels we already ship. Only the small IQ1_M router /
    hc matrices would stay FP16 (they are tiny; "small shapes stay FP16" is also a1444e33's rule).
  - a1444e33 measured: IQ3_S 2K prefill −27..−30%, 16K −16%, follow-up turns −6.5..−22.7%, KL
    +1.8e-4, argmax agreement unchanged (**known** T11.R1 #1; our pack geometry matches theirs).
- **Prefill expert streaming** (the 6.5-7.0k t/s plateau): misses/all-experts go through the
  Stager — a **16-slot** pinned ring (prefill.cpp:427 `int kRing = 16;`, STRATA_STAGER_RING
  override 2..256 :453) fed by N copy threads (:490-541), then **one cudaMemcpyAsync per blob**
  on the copy stream (:2362, :2441). STRATA_DMA_BATCH does NOT apply here (it wraps only the
  verify fetch verify.cpp:2610 and the adapt-tier copies generate.cpp:9076/:13967) — that is
  ~512 blobs/layer x 48 layers ≈ 24.6k per-blob DMA submissions per 2048-token chunk. T12-MEM #7
  found the long-context decay is staging-SYNC, not bandwidth; the same machinery is the natural
  suspect for part of the flat plateau. The instrument already exists: STRATA_PREFILL_TIMING
  (prefill.cpp:2097 host_sync_ms/host_chunk_ms/host_setup_ms).
- PLE projections still on the critical path at layer 1 (2d6b1db8 PLE-beside-layer-0 port,
  **known** T11.R1 #2; verify.cpp `l == 1 && ple_on` confirmed in the prefill flow).
- 95979cf2 profile-ordered copy-ahead in prefill: **known** T11.R1 #3; T12-MEM #7 correctly
  bounds its value to mid-size chunks (at 32K chunks all experts route, so reordering is inert).

### 1.7 Where do prefill and cache residency still leave t/s on the table? (the explicit question)

- Single-stream 250: NOT in the cache (97.7% hit, VRAM-hits line at 87-100% roofline — T12-MEM
  #1/T12-GPU both confirm; expert bytes are done). It is in (i) acceptance depth (FastMTP retrain,
  **known** GO T11.R3; EcoSpec tree filling **known** T9.B#1), (ii) the small-kernel latency lines
  (attention pre75 port **known** #1; hc ALU-bound line **known** T12-MEM #2; commit/drafter
  round), and (iii) — NEW this audit — three landed-but-unarmed env arms (below) worth one
  battery each before any new code is written.
- Conc 300: the pool-overhead package (T12-HOST) + the R2 cohort pipeline (**known**) remain the
  levers; this audit adds that two of T12-HOST's items are already coded (diet) or trivially
  reachable (miss-list), and that the DMA-batch treatment should be extended to the prefill
  stager and audited for the conc lane's staging share.

---

## PART 2 — RANKED IMPROVEMENT CANDIDATES (value x certainty / cost; known items marked)

**#1. Arm the three landed-but-off env arms: STRATA_POOL_DIET=1, STRATA_FUSE_HITS=1,
STRATA_FUSE_WAIT=1** — **known mechanisms, NEW status: built, never armed.**
Mechanism + evidence: pool.cpp:755-759/:901-904 (skip trailing repark; epoch claim fences
stragglers); verify.cpp:284-303 + native_moe.cu:46 (moe_hit_add folded into combine, ZADD
bitwise); verify.cpp:262-269 (fused wait+copy doorbell, ~144 nodes/window, F-204 scan).
Expected impact: T12-HOST's own estimate for the diet −0.5..1.0 ms exposed/window = +2-4% on
pool-exposed lanes; the two fuses −96..~300 nodes/window, and F-206 warns node savings ≠ wall
time — call it +0-2% each, honestly unknown until measured. Cost: S (one ABAB battery each,
battery-config.sh env override). Risk: diet = scheduling-only (bits unchanged; canary boot);
fuses = claimed bit-identical (M3b canary must confirm 0 divergent tokens). Confidence: medium.
Cheap falsification: one ABAB each; diet regression would show as straggler wakeups in
STRATA_POOL_TRACE.

**#2. a1444e33 dense-MMQ prefill on CUDA — re-priced DOWN in cost** (**known** item, NEW
evidence). Mechanism: lift `__HIPCC__` gate at gemm.cu:812/:936 and allow STRATA_PREFILL_MMQ
on CUDA (CMakeLists.txt:72/:88-89); keep STRATA_DENSE_MMQ as the arm; Q6_K MMQ instances already
in strata_kq (mmq_kq_lib.cu:53-56); small IQ1_M router/hc shapes stay FP16 (refusal semantics
already in gemm.cu:828). Expected impact: eddoursel measured −27..−30% @2K, −16% @16K; on our
6.5-7.0k t/s plateau that is roughly +25-40% prefill t/s (8.5-9.5k class) with zero decode
effect; priority metric with no bigger queued lever. Cost: M (gate + wiring + parity harness;
smaller than the "~130 lines" if Q6_K covers the bulk — verify by listing dense shapes x types
from the artifact first). Risk: different bits (q8_1 activations vs FP16-dequant cuBLAS) →
M3b canary + 2x MMLU-Pro mandatory. Confidence: high (measured on the sibling fork; same pack
family). Cheap falsification: run the 1783-token prefill probe (campaign/prefill-probe.sh) with
a locally-built arm; no double-digit % ⇒ stop.

**#3. FastMTP head retrain** (**known** T11.R3, verdict GO — restated for completeness, not
re-derived). Mechanism: recursive teacher-forced CE on the MTP head, self-distilled 5k
conversations, bf16 → direct-cast Q4. Impact: T11.R3 decision math ~227 → ~237 → 247-255 with
depth re-unlock = THE single-250 path. Cost: L (first training run; 1-2 days eng + weekend
compute). Risk: draft-only; output-safe by construction (verify untouched); gate = T11.R3's
pos-2/3 +≥8pp, pos-0 ≤1pp, e2e ≥+8%. Confidence: high. Falsification: the 15-30 min 9B-sibling
dry run.

**#4. EcoSpec expert-cost-aware tree filling + marginal-cost depth** (**known** T9.B#1).
Mechanism: score draft-tree nodes S = P/ΔCost, ΔCost = new HBM experts forced; land in
src/spec/draft_policy.cpp (exists — verified this session); our router gates are already
materialized (usage/credit counters, expert_source.cpp:3731-3733) so the predictor is ~0 ms.
Impact: 1.36x vs EAGLE-3 1.22x on Qwen3-235B from node selection alone; est single +12-20,
conc +8-12. Exactness-safe (candidates only). Cost: M. Confidence: medium. Gate: differential
nospec + acceptance monitors. Falsification: from current logs, count windows where the chosen
tree forces ≥1 new expert — if <20%, ΔCost never binds.

**#5. ROUTE_RESIDENT margin sweep + admission-reordering A/B** (**known** T12-MEM #5, extended
by the audit's no-eviction finding). Mechanism: 0.75→{0.65,0.85} env sweep; then one admission
policy A/B (upstream #2458 LRFU vs tqllm decayed-admission vs REAP-importance placement
**known** T4.3) — all now provably *admission* changes (1.1). Impact arithmetic: +1 hit-pt =
−43.5% miss bytes at 7% miss (T7.C.3); at our 2.3% miss the absolute is small: +2-4% decode
while the pool is serial, plus a flatter conc tail. Cost: S (sweep) / M (policy port). Risk:
routing-changing → 2x MMLU-Pro. Confidence: medium. Falsification: one battery per margin;
metrics hit-rate counters already report the answer.

**#6. STRATA_DMA_BATCH for the prefill stager** (NEW, code-internal). Mechanism: route the
stager's per-blob cudaMemcpyAsync (prefill.cpp:2362/:2441) through `copy_expert_blobs` /
dma_batch_mode() — the exact wrap verify.cpp:2610 already uses; optionally raise the 16-slot
ring (STRATA_STAGER_RING) to cover batch latency. Evidence: F12 measured 84.9% fewer DMA submit
cycles on the decode path; T12-MEM #7 fingered staging-SYNC as the long-ctx decay. Impact:
+0-4% prefill (bounded by the host/stager share of a chunk). Cost: S. Bits: identical (same
bytes, same order). Confidence: low-medium. Falsification: STRATA_PREFILL_TIMING=1 (prebuilt at
prefill.cpp:2097) — if host_sync+chunk < 2% of chunk time, retire without writing code.

**#7. hc bf16-native op variant in fused_gr** (**known** T12-MEM #2 lane; audit confirms the
line is ALU/cvt-bound, not byte-bound: 3.93/4.00 ms heavy vs 26.2 MB/layer). Mechanism: keep the
bf16 hc weights packed through the Mix/Combine math (kill the unpack8 cvt chain) instead of the
falsified wide-grid (F-217). Impact: halving the 2.4-4.0 ms hc line = −1.2..2.0 ms/window ⇒
+2-5% decode. Cost: M. Risk: different bits → M3b + 2x MMLU-Pro. Confidence: medium.
Falsification: ncu on fused_gr — if cvt/ALU pipes are NOT the stall source, stop.

**#8. QSA prefill attention batch (#2756 port)** (**known** T8.C; code-verified absent).
Mechanism: 32→128 query batch in the QSA prompt-attention pick. Impact: ~9% of QSA prompt-attn
time; QSA-prompt-attn is a minor slice of the chunk ⇒ +0.5-2% prefill. Cost: S-M. Bits:
upstream claims bitwise via FP32-hash parity — re-verify with our parity harness anyway.
Confidence: medium. Falsification: profile qsa_prompt_attn share on a 32K probe before porting.

**#9. Bole-class factor-tree GDN verification** (NEW external; arXiv 2608.01651). Mechanism:
encode each draft node's recurrent-state update as token-level factors (the delta rule's rank-1
correction), verify the whole tree in parallel, reconstruct only the sampled path — SGLang-
integrated, 3.4-7.7x linear-attention tree verification, up to 2.03x over the strongest tree
baseline, inference-only. Maps to us: YES structurally (hybrid GDN + tree drafts on the MTP
head) — it is the published generalization of the T7.B slot-per-draft design and of F-216's
slot-promotion. BUT our own F-216 result ("replay is NOT the depth gate") caps the expected
value: it pays only if the stair diagnostic unlocks depth 5-8 at ≥70% acceptance. Impact:
+8-15% IF unlocked (same arithmetic as T6#5). Cost: L (kernel + verifier surgery). Risk:
different bits (reassociation) + window-structure change → full gate incl. the F-191-class
re-test of PDL-family assumptions. Confidence: low. Falsification: the stair diagnostic first
(**known**, cheap); depth never offers ⇒ dead.

**#10. Draft-horizon expert prefetch via the MTP window** (partially **known** T8.A#3; NEW
mapping evidence MoE-SpeQ arXiv 2511.14102 + DuoServe 2509.07379). Mechanism: the coupled draft
tokens are known before verify — route their predicted chain through layer routers to prefetch
window misses into staging during the previous layers' compute. Audit finding that changes the
plan: ForesightSwap (the built prefetch tier) is force-disabled under DEVICE_PLAN
(expert_source.cpp:5256-5260) ⇒ the work = attach prefetch to the staging/adapt machinery or
reconcile the two. Impact: bounded by the 2.3% miss rate ⇒ +1-2% decode, conc a bit more
(miss clustering). Cost: M. Risk: prediction-only (output-safe). Confidence: low-medium.
Falsification: log per-window misses' advance-notice distance; if misses are already staged a
layer early, there is no headroom.

**#11 (mention-only, no rank). PreScope (arXiv 2509.23638) learnable deep predictor** — our
RouterLookahead already applies the ACTUAL next-layer router (oracle-grade at depth 1); a
learned predictor only wins at depth ≥2 where we deliberately degrade (set_depth ≤4). Not worth
it while depth-1 warms are free. MoE-Infinity (2401.14361, activation-trace prefetch),
Mixtral-offloading (2312.17238, LRU + speculative prefetch), EdgeMoE — all dominated by the
exact-router lookahead we already run; MoE-SpeQ's AcceptMoE = EcoSpec-class (**known** T9.B#1).
Gated DeltaNet-2 (2605.22791), Expert Streaming (2603.27624, multi-GPU), Intel AI-PC ultra-low-bit
(2508.06753, Xe kernels), Scale-Folding sub-4-bit (2603.17891, re-quantization = TurboQuant
SKIP class **known**), HybridGen (2604.18529, CPU-GPU attention) — map: NO (training-side,
multi-GPU, or re-quantization). Blackwell: tcgen05/TMEM sm_100-only re-confirmed for sm_120
(Colfax, gau-nernst, NVIDIA forums — corroborates **known** T7.A/T9.A traps); TMA IS available
on sm_120 but decode-GEMV TMA stays a loss (**known** T9.A trap); 2:4 sparsity = repack costs
bytes at bs1 (**known** T7.A do-not-bother); PDL closed by F-191; clusters cap 8 portable
(**known**). Megakernel family: Hazy no-bubbles / Mirage MPK **known** (T7.A G1, T8.A MPK-lite);
Ada-MK (2605.11581) and AutoMegaKernel (2606.09682) are automated megakernel compilers — same
do-not-bother verdict until a data-dependent-MoE-with-offload instantiation exists anywhere.

---

## PART 3 — CONTRADICTIONS / LEDGER CORRECTIONS vs the given context facts and our own records

1. Context said "foresight/swap tier (d.fs->take/note_miss ~3736+)" as a live mechanism — TRUE in
   code, but it is force-disabled under the standing STRATA_VERIFY_DEVICE_PLAN=1
   (expert_source.cpp:5256-5260). Dead code in the standing recipe (**known** as a note; the
   contradiction is that banked queue items T8.A#3/DuoServe-prefetch implicitly assume it runs).
2. The SVG-CUSTOM-TODO T12-HOST ledger is stale on two items: #1 (plan one-pass) is LANDED and
   default-ON with an independent diff auditor (expert_source.cpp:3739-3827); #2 (barrier diet)
   is LANDED as STRATA_POOL_DIET but un-armed (pool.cpp:480-481). T12-GPU item 4's "why is
   hits_added gated off" is answered: STRATA_FUSE_HITS unset on CUDA (verify.cpp:303).
3. T12-MEM/T11.R1 priced the a1444e33 port "~130 lines + 5 sites"; the audit adds the load-bearing
   fact that the standing pack's dense GEMMs are predominantly Q6_K and our strata_kq already
   builds Q6_K MMQ instances (mmq_kq_lib.cu:53-56) — the port's kernel half may already exist.
   (Also: our tree refuses non-256-multiple K rather than zeroing tails, gemm.cu:822-828 — the
   "tail-zeroing gotcha" is currently handled by refusal.)
4. All other context facts verified: 36 GDN + 12 QSA (GGUF: block_count 48, full_attention_interval
   4 ⇒ 12 QSA), expert_count 512 / used 10, n_embd 2560, ff 640, indexer top_k 2048 (GGUF header,
   this session); 2.046 MB avg blobs / 9,337 slots / 38% / 97.7% (T12-MEM ledger); prefill plateau
   with no decay 32k-120k (T12-MEM #7: decay begins 157k→824k, staging-SYNC); F-224 adopted
   (journal + launch line 9); F-184/F-200/F-201/F-221 standing arms (launch file read).

## OPEN QUESTIONS

- Dense tensor list x K-dimension coverage for the MMQ arm (which of attn_qkv / attn_output /
  ssm_out / output.weight have K % 256 != 0? output.weight N = vocab 248,044 is N-side, fine).
- Whether the conc lane's GPU-reach wait includes prefill-stager sync (STRATA_PREFILL_TIMING on
  a conc probe would answer).
- kStagingBlobs 32→48: does the DMA pick ever saturate 32 in conc windows (counter exists in the
  fetch path)?
- FUSE_WAIT/FUSE_HITS have never been A/B'd on CUDA — the "bit-identical" claims are code
  arguments, not measurements.

## METHODS NOTE (so nothing is repeated)

Unproductive/low-yield searches: "MoE-Infinity" (dominated by our exact-router lookahead);
"Expert Streaming 2603.27624", "HybridGen 2604.18529", "Intel AI-PC 2508.06753" (multi-GPU /
Xe / out-of-scope); PreScope (interesting, but depth-1 oracle already free); generic
"survey MoE offload 2025" (no dedicated survey; awesome-moe-inference list is the index).
Productive: Bole, MoE-SpeQ, tcgen05-vs-sm_120 corroboration, and the GGUF header parse (dense
types) — the last was the single highest-value external+internal cross-check of the session.

## SOURCE LIST

Code (all read 2026-10-16, tree @ F-224): expert_cache.cpp:622-656/:69/:374-380/:382-445;
expert_cache.hpp:108/:170-171/:237; expert_source.cpp:3739-3827/:3867-3945/:3997-4002/
:3345-3428/:5014-5277; expert_source.hpp:155-239/:319-341/:360-498/:576-624/:916-1035;
pool.cpp:24-43/:478-481/:735-759/:845-904; pool.hpp:239-242; verify.cpp:90-303/:700-791/
:920-930/:1499-1584/:1688-1735/:2596-2630; verify.hpp:316-317/:548-556; gemm.cu:1-9/:640-721/
:813-951; mmq_kq_lib.cu:53-56; qsa_prompt_attn.cu:34/:1455; prefill.cpp:35-41/:422-541/:560-640/
:2060-2178/:2323/:2362/:2441/:2736/:2918-2922; prefill/kernels.cu:600-688/:714-1063;
generate.cpp:488/:677-709/:6863-6990/:9076/:10831/:13967; native_dense.cpp:31/:143-157;
CMakeLists.txt:72-89/:79/:1275-1276/:1437-1447; launch-custom-iq3s-8081.sh:8-36;
custom-iq3_s-8081.json; battery-config.sh; campaign/research/SUBWARP-EXPERT-PORT-PLAN.md;
artifact GGUF header (Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001, parsed this session).
Campaign records: SVG-CUSTOM-TODO.md (T1-T12 folded), SVG-CUSTOM.md F-179/F-184/F-191/F-196/
F-200-F-208/F-211-F-224, campaign/research/F-2xx agent dossiers.
External (new this session): arXiv 2608.01651 (Bole; fetched); arXiv 2511.14102 (MoE-SpeQ);
arXiv 2509.23638 (PreScope; fetched); arXiv 2401.14361 (MoE-Infinity); arXiv 2312.17238
(Mixtral-offloading); arXiv 2605.22791 (Gated DeltaNet-2); arXiv 2605.11581 (Ada-MK); arXiv
2606.09682 (AutoMegaKernel); arXiv 2509.07379 (DuoServe-MoE — **known** T7.C.2, re-cited);
research.colfax-intl.com (Blackwell clusters/tcgen05), gau-nernst.github.io/tcgen05,
forums.developer.nvidia.com (sm_120 support), hazyresearch.stanford.edu no-bubbles + Mirage MPK
(**known**), docs.nvidia.com Megatron gated_delta_net (corroborates our WY form), sustcsonglin
DeltaNet-2 blog. Known-external carried by citation only: FastMTP 2509.18362, EcoSpec 2607.12696,
Fiddler 2402.07033, KTransformers SOSP25, T-MAC, ProMoE/MoEpic/FluxMoE, HOBBIT, QuantSpec,
llama.cpp #29887/#29353/#30087/#24528/#2458, upstream #2756/#2612/#2746, eddoursel fork commits.
