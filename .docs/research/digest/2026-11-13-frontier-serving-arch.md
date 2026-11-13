# Frontier serving-architecture audit (2026-11-13) — vLLM V1 / SGLang / TRT-LLM / LMCache vs our serving stack

Batch 1 of the 18-agent frontier wave (operator-ordered full research gathering). Phase A: frontier docs/blogs/PRs
with numbers. Phase B: direct audit of serve/server.py (9,066 lines), src/program/generate.cpp (18,533),
src/core/verify.cpp (4,721).

Correction established: kVerifyMaxT=8 means "per-width capture" is 8 graphs = an effective bucket-per-width
scheme already; the doorbell loop is (le_-lb_)*G = 48-96 host round trips per window.

## Ranked findings
1. **Per-layer-group doorbell host spin vs frontier zero-sync** (verify.cpp:3141-3219). vLLM V1 async
   scheduling / SGLang zero-overhead scheduler (1.1x measured). In-camp plan already named: T1.1 partial
   device plan → plan-thread decoupling → batch G=2. F-208 attributes the tail: 4.90 ms CPU-expert service
   serialized by a SINGLE service thread; batch windows forfeit the group-split.
2. **No overlap scheduling in the serve lane** — batch_step collects serially; cohorts ARE overlap but
   degenerate below n_active > batch/2 (generate.cpp:12288); first cohort round even does a
   cudaDeviceSynchronize (12293).
3. **No radix prefix caching — reuse is slot-identity-bound** (pick_slot longest-prefix + turn checkpoints;
   cross-request zero reuse). Frontier: vLLM V1 default-on <1% cost; SGLang radix 1.9x throughput/3.8x hit;
   CacheBlend 2.2-3.3x TTFT. Intermediate for us: hash-mount ConvCheckpoint chains across ANY slot
   (generate.cpp:9705-9804) before any radix tree.
4. **Chunked prefill: fixed ladder + cadence (default OFF) vs Sarathi token-budget mixing** (OSDI'24: up to
   6.65x throughput class). Our mixed-row vocabulary exists (server.py:327 comment).
5. **Batch windows forfeit G=2** (verify.cpp:1235 `batch_rec_` forces G=1; the STRATA_BATCH_G2 arm is
   captured but the batch service loop reads one group) — the cheap graph-side flip, ordered after plan-thread
   decoupling by F-208.
6. **KV does not tier** (experts do). LMCache 7x KV access/100x capacity; Mooncake >95% hit. Extend
   ConvCheckpoint to turn-granularity + SSD spill.
7. No KV importance-eviction/compression (tail-slot eviction only). QSA/GDN state differs from plain K/V —
   rank behind 6.
8. CUDA graphs: 8-width capture fine; the gap is the SHARED GRAPH MEMORY POOL (SGLang one-pool-across-
   captures; breakable graphs 5x builds/1.7x prefill).
9. **Prefill runs the same per-layer doorbell** — breakable-graph/piecewise direction applies literally;
   shares finding 1's rewrite (doorbell_publish_res already miss-aware).
10. Residual per-step host syncs (2ms stream query, cohort-transition device sync, BT flushes) — cheap deletes.
11. Admission count-based not token-budget-based — minor fairness.
12. NUMA locality of doorbell-serviced rows vs pool cores unverified (ds4_os has mmap_on_node; bind service
    thread to the rows' node; KTransformers/HOBBIT precedent).

## Already matching or beating the frontier
Cohorts (overlap inside one instance — ahead of curve, gap only below the engagement threshold); conversation
cache + T66 persistence (per-slot frontier-grade); swap-guard safety kit; prewarm lifecycle (frontier ships
nothing comparable); boot pre-capture of all 8 widths + commit; async window commits (F-196); elastic VMM KV
(vAttention-class, addresses never move — why graphs stay valid); expert memory tiering (HBM→pinned→DRAM→
page-cache implemented); NUMA/hugepage/hybrid-core machinery; the ar_on() all-resident zero-sync path.

## Recommended order
5 (G=2 flip, cheap) → 1+9 (whole-window service thread + device plan, the named rewrite) → 4 (cadence on,
then row budget) → 2 (always-split cohorts + decoupled admission) → 3 (cross-slot checkpoint mount) → 6 → 8 →
10-12 hygiene.

Sources: vLLM V1 blog; vLLM CUDA graphs design; AWS Neuron async scheduling; Berkeley EECS-2025-192; SGLang
v0.4 + advanced CUDA graphs + piecewise docs; Sarathi-Serve OSDI'24; FlashInfer arXiv:2501.01005; TRT-LLM IFB;
LMCache blog+benchmarks; CacheBlend arXiv:2405.16444; KV survey arXiv:2607.02574; llama.cpp NUMA discussion
#19102; KTransformers SOSP'25; HOBBIT arXiv:2411.01433.
