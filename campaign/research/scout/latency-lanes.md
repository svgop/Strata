# Scout Digest — Latency Lanes (TTFT p99, ITL p99 @ 6-concurrent, single RTX 5090, MoE + MTP)

Date: 2026-10-25. Scope: implementation-grade scheduler mechanics from 6 comparable stacks.

---

## 1. vLLM V1 scheduler — `vllm/v1/core/sched/scheduler.py`, `vllm/config/scheduler.py`

**Code-level mechanics:**
- Two budgets per step: `token_budget = max_num_scheduled_tokens` (defaults to `max_num_batched_tokens`) and a separate `input_budget = max_num_batched_tokens`. With spec-dec, every scheduled request charges `input_budget` by `num_new_tokens + draft_slots` where `draft_slots = spec.max_num_new_slots_for_drafting`; guard `if input_budget <= draft_slots: break`. Draft tokens count against the budget — direct analog for our MTP window.
- Decode-first: RUNNING requests scheduled first; waiting (prefill) loop runs only `if not preempted_reqs`. `long_prefill_token_threshold` clips a prefill to the cap, **waived when the request is the only one in the batch** (`if num_eligible_reqs > 1 else 0`) — explicit anti-starvation carve-out for TTFT.
- Chunking: `num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)`. `_reserve_prefill_lookahead` avoids ending a chunk within `num_prefill_lookahead` tokens of prefill end (multi-module MTP drafter constraint).
- Preemption: on `new_blocks is None`, victim = `self.running[-1]` (FCFS) or lowest priority; **one request at a time** in a `while True` retry loop; RECOMPUTE mode (`num_computed_tokens = 0`).
- Spec: decode steps padded to `1 + num_spec_tokens` only when `not prefill_scheduled` — keeps the step decode-only for a **full CUDA graph**; mixed prefill+decode steps drop that padding. `skip_drafts` skips K=0 draft passes on non-final prefill chunks.

**Defaults (real numbers):** `DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` / `DEFAULT_MAX_NUM_SEQS = 128` in `config/scheduler.py` (comment: "mainly for convenience when testing… real usage set in `EngineArgs.create_engine_config`"); per docs/issues the V1 production default is **8192** for decoder LLMs (16384 on large-memory GPUs; 16384+ recommended for pooling), `long_prefill_token_threshold = 0` (disabled), chunked prefill `True` always, preemption mode RECOMPUTE. Docs: "It batches all pending decode requests before scheduling any prefill"; smaller budget (2048) → better ITL; `>8192` → throughput.

**Transfers to us:** budget split (decode charge 1(+K) tokens, prefill fills remainder), single-request TTFT waiver, draft slots in the input budget, full-CUDA-graph decode-only steps. **Doesn't transfer:** RECOMPUTE preemption (wasteful for 6-slot p99 — we should never need it at 6 concurrent); 8192-budget throughput orientation.

## 2. SGLang — overlap scheduler + PrefillAdder (`managers/scheduler.py`, `managers/schedule_policy.py`, `arg_groups/fields/schedule.py`)

**Defaults:** `chunked_prefill_size` field default `None` (-1 disables; resolved at runtime — maintainer: "by default sglang uses chunked prefill with a chunk size of 8k", 4096/2048 recommended on OOM/smaller GPUs); `max_prefill_tokens = 16384` ("real bound = max of this and max context"); `disable_overlap_schedule = False` (overlap ON); `schedule_conservativeness = 1.0`; `max_running_requests = None`.

**Mechanics:**
- Overlap loop `event_loop_overlap()`: launch batch N, append `(batch, result)` to `result_queue`, then `pop_and_process()` results of N-1 while N runs on GPU. WAR barrier pins scheduling writes behind the forward's reads (`schedule_stream.wait_event`). Env `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=False`: **two consecutive prefill batches disable overlap to improve TTFT of the first**. Grammar + spec-v2 barriers force overlap off when FSM sync pending.
- Admission (PrefillAdder): KV budget is an object `create_prefill_budget(...)`; each admitted request charges `extend_input_len + max_new_tokens*ratio (+ mamba_gap_reserve + request_slot_reserve)`, with `max_new` clipped by running-offset `min(max_new_tokens - len(output_ids), CLIP)*new_token_ratio`. Failure → `AddReqResult.NO_TOKEN`. Radix nodes pinned during admission via `_lock_node(req.last_node)` ("temporary pin excludes this prefix from the evictable budget") — prefix-hit requests are admitted against a budget that already discounts the shared prefix.
- Chunk split: chunk is page-snapped (`extend_len // page_size * page_size`, end aligned to `(prefix+len)//page*page`); `max_new_tokens = 0` for the chunk step; **one chunked req at a time**. Decode "charges" are mixed into the prefill batch head (`rem_input_tokens -= num_mixed_decode_tokens`).
- Retraction (their preemption): `retract_all` releases KV in order; eviction headroom `remaining_reqs * SGLANG_RETRACT_DECODE_STEPS` (`EnvInt(20)`); `reset_for_retract` clears `prefix_len=0` → full recompute on re-admission; disagg mode optionally offloads KV to host for restore-without-recompute.

**Transfers:** overlap of CPU schedule with GPU step (our "async admission"); page-snapped chunk ends (cheap CUDA-graph reuse); reserve-per-request charge incl. draft tokens; consecutive-prefill-overlap-off for TTFT. **Doesn't transfer:** retraction-as-preemption philosophy (we cap concurrency at 6 so budget should guarantee fit); `schedule_conservativeness` ratio heuristics only matter at unbounded queue.

## 3. TensorRT-LLM PyTorch — CapacityScheduler (`tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`)

**Mechanics:**
- `CapacityScheduler.schedule_request(active_requests) -> (scheduled, paused)`. Default binding: `BindCapacityScheduler` wrapping the C++ impl, `scheduler_policy = GUARANTEED_NO_EVICT`, `enable_prefix_aware_scheduling=True`, scheduling window `[CONTEXT_INIT, GENERATION_COMPLETE)`.
- **GUARANTEED_NO_EVICT admission test:** per request, reserve blocks-to-completion: `NoEvictScheduledBlocksManager.enough_available_blocks(req, summary)` (dual pools both must pass); new request admitted only if its *entire* completion fits; on first failure: `break` (strict head-of-line, no reorder). Radix summary (`PrefixReuseSummary`) is threaded in so "the C++ side can skip a redundant radix-tree walk."
- **MAX_UTILIZATION alternative:** optimistic `prepare_blocks_if_schedulable`; on failure it **pauses the newest started request** (`paused_req = requests_list[last_started_idx]`), `scheduling_remove_sequence(id)` frees its KV, retry; `paused_requests` = "KV cache suspended without resetting request state" — pause/resume without recompute.
- Context chunking lives in `PyMicroBatchScheduler`: token budget `if max_num_tokens is not None and (batch_num_tokens + compute_tokens > max_num_tokens): break`, then `remaining_capacity = max_num_tokens - batch_num_tokens` and `_set_ctx_requests_chunk_size(contexts_to_be_chunked, remaining_capacity)` — i.e., **chunk size is derived from leftover budget, not a fixed chunk**. KV-reuse-aware: compute = `max(0, remaining - reusable)`.

**Transfers:** reserve-to-completion admission is exactly our "predictive parking restore" precondition (admit only when the parked slot's KV + its remaining N tokens provably fit); pause-without-recompute = park with KV retained; leftover-budget chunk sizing. **Doesn't transfer:** GUARANTEED_NO_EVICT's strict FIFO `break` (can starve TTFT behind a huge reservation) — take the test, not the ordering.

## 4. llama.cpp `llama-server` — continuous batching via ubatches (`tools/server`)

**Defaults:** `-np` slots `-1` (auto); continuous batching **enabled by default**; `-b` n_batch = **2048** (logical), `-ub` n_ubatch = **512** (physical); `--cache-prompt` on; `--cache-reuse 0` (off; KV-shift reuse min chunk); `--slot-prompt-similarity 0.10` slot affinity; `--kv-unified` on when slots auto.

**Mechanics:** prompts are split into 2048-token logical batches → each chopped into ≤512-token ubatches ("a 5000-token prompt becomes … ten forward passes: four, four, then two"). `split_simple()` fills ubatches "in submission order… does not look at sequence ids" — with unified KV, **one ubatch mixes prefill rows of one slot with decode rows of other slots**, so every active conversation gets a row per step and a "one-token decode really is a one-row matmul" (no padding). Decode stall avoidance is therefore structural: no slot's prompt ever runs as one monolithic kernel; the max uninterrupted chunk is `n_ubatch`. KV-cache length is padded to multiples of **256** "so the graph can be reused — only every 256th step pays for a rebuild." Caveat: "verifying a five-token draft under -ub 4 costs two passes" — ubatch must exceed draft window (K+1). No preemption: slots run to completion; requests queue (`requests_deferred` metric).

**Transfers:** our "interleaved bounded prefill" ≈ their ubatch: **bound the max contiguous prefill slice so a decode lands at least once per slice**, and quantize KV growth to a graph-reuse quantum. **Doesn't transfer:** no admission test at all (memory planned per-slot at startup); single-node greedy slot selection.

## 5. Single-GPU local stacks

**Ollama:** not a batch-scheduler study — concurrency knobs only: `OLLAMA_NUM_PARALLEL` auto → **4 or 1** by free memory (batched sequences in one runner context), `OLLAMA_MAX_LOADED_MODELS` = 3×GPU count, `OLLAMA_KEEP_ALIVE` = **5m**, `OLLAMA_MAX_QUEUE` = 512. Lesson for 6-slot p99: pick a fixed, memory-verified concurrency (their auto=4 on big GPUs shows 4–6 fits typical 24–32GB) and stop unloading/drain-thrash under load (5m keep-alive defeats cold-start TTFT spikes).

**TabbyAPI / ExLlamaV3:** `chunk_size = 2048` ("amount of tokens per chunk with ingestion; lower reduces VRAM at cost of ingestion speed"), `cache_size = max_seq_len` (KV per-sequence, not pooled), `max_batch_size = None` (auto-clamped by cache). Ingestion chunking is VRAM-oriented, not ITL-oriented: exllama-style generators still drain decode during ingest chunks, so ITL p99 under 6-concurrent is dominated by the 2048 chunk — a counter-example confirming our decode-first budget.

**LM Studio:** closed-source (llama.cpp core) — inherits the ubatch story; no new knobs.

## 6. Sarathi-Serve (microsoft/sarathi-serve, public, vLLM fork) — the token-budget calculator

**Mechanics (`sarathi/core/scheduler/sarathi_scheduler.py`):** Phase 1: all running decodes get `num_batched_tokens += 1` each (`_append_slot`), slot-preemption only `while not block_manager.can_append_slot()` (victim `running.pop(-1)`). Phase 2: continue partial prefill then new prefills FCFS, `next_num_tokens = min(prompt_remaining, chunk_size - num_batched_tokens)`; stop at `max_num_seqs`. **Fixed static token budget for the whole session** (paper: budget 512–2048; 512 their latency pick; paper's stall-free claim conditioned on TPOT < 10 ms — our exact ITL target). Key formula: chunk selected so step latency satisfies TBT SLO — hybrid step `T ≈ T_batched_non_attn + T_attn(reqs) + T_overhead`. Dynamic mode: `np.linspace(low_chunk_size, high_chunk_size, stages)` reversed, rounded to multiple of 32, per-stage budget `ceil(chunk_schedule_max_tokens / stages)` — a **staged adaptive chunk window** (their "dynamic split-wise" = our per-cohort adaptive draft window analog, but for chunk size).

**Transfers:** static budget sized to the ITL SLO (start 512–1024 for MoE on 5090); decodes-then-prefills ordering; 32/256-multiple alignment for graph reuse. **Doesn't transfer:** their +1-token decode charge assumes no spec-dec — we must charge `1+K` per slot (vLLM-style draft slots) or the budget lies.

---

## KNOB SHEET

| Knob | Their default | Our analog | Suggested first value |
|---|---|---|---|
| Per-step token budget | vLLM 8192 (prod) / 2048 (ITL); Sarathi 512–2048 | `step_token_budget` | **512** (charge 1+K per decoding slot; raise to 1024 only if TTFT p99 misses) |
| Prefill chunk cap | SGLang chunk 8k→4096; llama.cpp ubatch 512; TabbyAPI 2048 | `max_prefill_slice` | **256–512, 256-aligned** (≤ graph-reuse quantum; > K+1) |
| Decode-first ordering | vLLM/Sarathi: all decodes before any prefill | same | all 6 slots' decode steps first, then prefill slices |
| Single-request TTFT waiver | vLLM `long_prefill_token_threshold` waived when batch has 1 req | same | waiver when no decode slot is waiting; cap 1024 otherwise |
| Consecutive-prefill overlap | SGLang disables overlap for 2 consecutive prefills (TTFT) | scheduler/CPU overlap | keep async admission ON; never overlap admission with first prefill chunk of a fresh request |
| Admission test | TRT-LLM GNE: reserve blocks-to-completion | park/restore admission | admit new req only if `free_kv ≥ prompt_blocks + remaining_N * (1+accept_rate)` |
| Preemption | vLLM RECOMPUTE; TRT-LLM MAX_UTIL pause (KV kept); SGLang retract (recompute, 20-step evict headroom) | park, never preempt | at 6 slots: no preemption; KV-retained parking only |
| Draft slots in budget | vLLM `input_budget -= num_new_tokens + draft_slots` | MTP window charge | charge K per decoding slot; `skip_draft` on non-final prefill chunks |
| Spec-dec padding | vLLM pad decode to 1+K only in decode-only steps | CUDA-graph batch shape | keep prefill slices out of decode-only graph steps |
| KV growth quantum | llama.cpp 256-pad for graph reuse | KV pad quantum | 256-token allocation multiples |
| Adaptive window | Sarathi `linspace(low, high, stages)` r32 | per-cohort draft window K | K ∈ {2,3,4} stages, accept-rate-gated, multiples aligned to graph set |
| Concurrency cap | Ollama auto 4; llama.cpp slots | n_slots | 6 fixed, admission-gated |

## OUR NEXT STEP (cheapest P-A implementation detail)

Implement **Sarathi-phase-1 + vLLM draft-slot budget**: per step, (1) schedule every decoding slot first, charging `1 + K_current` tokens each to `step_token_budget=512`; (2) fill the remainder with prefill slices capped at 256 tokens, 256-aligned, page-snapped, one partially-chunked request at a time; (3) waiver: a lone request's chunk cap lifts to 1024 when no decode slot is starved. This is ~1 scheduler function (no kernel work), reuses existing CUDA graphs for decode-only steps, and directly attacks both p99s: ITL p99 stops seeing prefill interruptions >256 tokens, TTFT p99 gets guaranteed forward progress every step. Instrument per-step `budget_remaining` + accept-rate to choose the K stage next.

---
### Grades
- vLLM V1: **A** (spec-dec-aware budgeting, decode-first, exact numbers) — sources: docs.vllm.ai/en/stable/configuration/optimization, github.com/vllm-project/vllm `vllm/config/scheduler.py`, `vllm/v1/core/sched/scheduler.py`, issues #18547/#60067/#4044.
- SGLang: **A-** (overlap + admission + retraction verified in code; resolved chunk default partly inference) — github.com/sgl-project/sglang `managers/scheduler.py`, `managers/schedule_policy.py`, `arg_groups/fields/schedule.py`, `environ.py`, discussion #1468, docs.sglang.io FAQ.
- TensorRT-LLM: **A-** (exact policies/pause mechanics; C++ probe internals not read) — `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`, docs/source/torch/scheduler.md.
- llama.cpp: **A** (ubatch interleaving fully explained with numbers) — tools/server/README.md, joelfernandes.org ubatch post.
- Ollama/TabbyAPI: **B** (knobs only, no ITL-specific tuning docs) — docs.ollama.com/faq, TabbyAPI wiki 02.-Server-options.
- Sarathi-Serve: **A-** (scheduler code verified; repo README carries no defaults — paper numbers 512–2048) — github.com/microsoft/sarathi-serve, arXiv:2403.02310, junuxyz/mlsys-notes.
