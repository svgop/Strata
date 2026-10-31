---
type: digest
id: w11-cold-scout
created: 2026-10-31
status: evidence
topic: W11 cold/jitter — warmup & CUDA-graph capture discipline in leading stacks
scope: read-only research; no code changes; sources fetched 2026-10-31
---

# W11 — Cold/jitter package: competitive warmup digest

Baseline under test: 2,361 lazy per-layout captures per boot (~10-40 ms each), boot pre-capture covers only serial-verifier window graphs; idle swap storm 96 swaps/round (promote 12-17 ms, park 24-52 ms); cache eviction forces 11.3 s cold re-read @ 80k tok; restart costs 42.7 s re-read @ 261k ctx. Fields per competitor: warmup mechanism, capture policy, persistence, measured numbers, grade.

## vLLM
- Warmup: memory profiling → KV alloc → dummy-run capture, all before serving; warmup = eager `_dummy_run` in `NONE` runtime mode, and "it is also important to explicitly run attention during the warmup dummy_run call".
- Capture: eager at init by default — 67 graphs over [1,2,4,8,16,24,...,512]; each captured size reserves its own buffer (stepwise memory scaling). `cudagraph_mode: NONE` / `enforce_eager` skips capture (faster boot, slower decode); `compilation_config.cuda_graph_capture_sizes` cuts the list.
- Lazy-capture RFC #20098 CLOSED not planned (Dec 2025): maintainer "I have a strong no for this feature" — runtime complexity, unpredictable OOM, perf dips; prescribed remedy = fewer capture shapes, not laziness.
- Numbers: startup capture "can take more than 10 s"; one reported boot 45 s with ~10 s capture (~20%); per-graph capture ~50 ms on an 8B model after optimization.
- Grade: verified-from-source (GitHub RFC thread + design doc).
- https://github.com/vllm-project/vllm/issues/20098 ; https://docs.vllm.ai/en/stable/design/cuda_graphs.html

## TensorRT-LLM
- Warmup: build-time kernel tuning layer by layer; timing cache (`--input_timing_cache` / `--output_timing_cache`, default `model.cache`) reuses profiling results across builds on same compute capability; `trtexec --warmUp` 200 ms default before any timing.
- Capture: PyTorch backend captures at runtime with warm-up iterations BEFORE capture — "kernel autotuning during CUDA graph warm-up" selects kernel variants. Explicit `cuda_graph_batch_sizes` list (real config: [1,2,4,8,16,24,32,64,128,256,320,384,512]) or `cuda_graph_config.auto_configuration` derives them from max_batch_size/max_num_tokens.
- Build: `--multiple_profiles` (default disable) = more min/opt/max shape profiles, better kernels, "increases engine build time but no other adverse effects" — the multi-batch-shape lever lives at build time, not runtime.
- Numbers: no published capture-cost figures; build time is the advertised cost. "Strongly-typed profiles" phrasing NOT confirmed on fetched pages.
- Grade: verified-from-source (build docs + issue #11791); auto_configuration fields medium (search-level).
- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html ; https://github.com/NVIDIA/TensorRT-LLM/issues/11791

## SGLang
- Warmup: capture phase before serving ("Capture cuda graph begin. This can take up to several minutes"), then a warmup generation request before ready; `--skip-server-warmup` (default False) skips it.
- Capture: decode = full graph per batch-size bucket; prefill = breakable CUDA graphs (default since v0.5.15) over token buckets — "a live batch is padded to the nearest captured token count"; request-slot padding is cheap (zero-length sentinels), token padding is the expensive dimension; unseen signatures fall back to eager. Bucket set is bounded — one experiment used 42 captured shapes; default capture <160/256, `--cuda-graph-max-bs-decode` raises to 512-768 at large TP (then reduce `--mem-fraction-static`).
- Numbers: torch.compile is 78-86% of prefill-graph prep time — 90 s on a 235B MoE, 158 s on GLM-5.2; breakable graphs build 3.8-5.2× faster (no compilation).
- Grade: verified-from-source (LMSYS blog 2026-08-17 + docs + issues).
- https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph ; https://docs.sglang.io/advanced_features/hyperparameter_tuning.html ; https://github.com/sgl-project/sglang/issues/18201

## KV persistence (LMCache / Mooncake — restart story)
- LMCache disk backend does NOT survive restart out of the box: metadata is "OrderedDict[CacheEngineKey, DiskCacheMetadata] in memory only"; after restart "new identical KV-Cache files are regenerated instead of reusing the existing ones" — TTFT equals cold start. Fixes: PR #2734 pickle dump/load of the metadata dict (cited Mar 2026); PR #4778 lazy filesystem recovery on lookup miss + self-describing `.meta` sidecar per chunk, "full-prefix reuse" at 17-24 GB/s retrieve across 4 models (issue reopened Aug 2026 — still maturing).
- Mooncake (vLLM connector): KV lives in MooncakeDistributedStore — external CPU pool, `master_server_address` :50051; hash-chain per block enables cross-instance/cross-restart lookup; `load_async` (default true) overlaps load with compute. Restart-survival implied (store outlives the engine), never explicitly documented. Both stacks require fixed `PYTHONHASHSEED` for stable block hashes.
- Numbers: LMCache GPU-cloud guide: 128k-ctx TTFT 11 s → 1.5 s on the reuse path; announcement claims 3-7× faster access vs recompute.
- Grade: verified-from-source for the LMCache restart gap (issue thread + maintainer comments); medium for Mooncake restart semantics (implied only).
- https://github.com/LMCache/LMCache/issues/1175 ; https://docs.vllm.ai/en/v0.21.0/features/mooncake_store_connector_usage.html ; https://github.com/vllm-project/vllm/issues/45036

## TGI (huggingface/text-generation-inference)
- Warmup: boot phase probes forward passes at increasing batch sizes to derive max feasible batch, capturing CUDA graphs along the way; VRAM shortfall aborts boot ("Please reduce --max-batch-total-tokens").
- Numbers: none published.
- Grade: medium (secondary anatomy article + issue #1300).
- https://github.com/huggingface/text-generation-inference/issues/1300

## llama.cpp (premise correction)
- The brief's "no capture — why it doesn't need it" is FALSE since Aug 2024: llama.cpp captures CUDA graphs by default for bs=1 decode; historically auto-disabled for bs>1 (PR #7302 removed unnecessary disabling); `GGML_CUDA_DISABLE_GRAPHS=1` opt-out; 2026 work extends capture to multi-slot decode.
- Why capture stays cheap there: ggml rebuilds one shape-stable compute graph per decode step → a SINGLE capture covers the whole session, no per-layout sweep. The transferable lesson is shape stability, not graph avoidance.
- Grade: verified-from-source.
- https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs

## What this changes on the row
- Every stack captures BEFORE serving; vLLM's lazy-capture RFC died with a maintainer "strong no" (OOM + perf-dip risk). Strata's boot-sweep plan is the orthodox move — keep it.
- 2,361 per-layout captures is an outlier: peers bound capture to a bucket ladder (~42 shapes SGLang, 67 sizes vLLM). Collapse strata's layouts into padded buckets first (token padding expensive, request-slot padding cheap — SGLang), or the boot sweep inherits the storm.
- Capture budget anchor: ~50 ms/graph (vLLM 8B) → 2,361 layouts ≈ 118 s swept naively; bucketing to <100 shapes keeps the sweep near SGLang/vLLM's seconds-scale budget.
- Restart re-read (42.7 s @ 261k) has a proven fix shape: LMCache PR #4778 — persistent sidecar metadata + lazy re-read on lookup miss, backgrounded (Mooncake `load_async` pattern). Pair it with strata's background file re-read rather than eager full reload.
- Stabilize hashes (PYTHONHASHSEED-equivalent) wherever conversation-cache keys derive from Python hashes — both LMCache and Mooncake restart bugs trace to unstable hashes.
- Gate the cold-3v3 battery on SGLang's discipline: warmup request must complete (and re-reads finish) before the first timed request; trtexec-style 200 ms warmup before measuring.
