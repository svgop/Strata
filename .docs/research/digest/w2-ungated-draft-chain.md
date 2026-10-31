---
type: digest
id: w2-ungated-draft-scout
created: 2026-10-31
status: evidence
---

# W2 — Ungated single-graph draft chain: competitor draft-loop structures

Question: how do leading engines run a multi-step draft loop without a host round trip per drafted token?
Field verdict: every leading engine keeps drafted tokens in device buffers between steps. The strongest two (SGLang, TRT-LLM one-model) bake the whole fixed-depth chain into ONE captured graph and trim post-hoc at verification. No leading engine reads the drafted token back on host between steps, and nobody early-exits the chain on device.

## TensorRT-LLM (PyTorch backend, v1.1.0 docs)
- Mechanism: two flavors. One-model (MTP / EAGLE-style): a `Worker` is "injected into the target model as a submodule"; docs state it "can launch the entire drafting loop as a single CUDA graph" — fixed depth, "does not support dynamic draft lengths". Supported pairs: EAGLE-3 + Llama-4-Maverick, MTP + DeepSeek-V3/R1. Two-model: drafting is host Python in the `PyExecutor` — `_prepare_draft_requests` attaches max draft tokens per request; `drafter.prepare_draft_tokens` fills `py_draft_tokens`, padded to max_draft_len ("a CUDA graph limitation" on the target capture).
- Termination: host, per round. One-model: the graph always runs the full chain; trim is post-hoc — greedy exact-match acceptance, then KV rewind (`py_rewind_len = allocated - accepted`, "very cheap since the blocks are just marked as available; no memory is actually freed").
- Token feedback: device buffers; target hidden states captured per layer via `maybe_capture_hidden_states` into pre-allocated buffers "for CUDA-graph compatibility".
- Graph boundaries: one graph over the whole draft loop (one-model); per-forward target graphs padded to max_draft_len (two-model). Two-model speculation disables the overlap scheduler.
- Measured overheads: none published. MTP offers relaxed acceptance (`relaxed_topk`, `relaxed_delta`) for thinking phases.
- Source: https://nvidia.github.io/TensorRT-LLM/1.1.0/features/speculative-decoding.html
- Confidence: high — official docs, direct quotes.

## vLLM (v1, main branch)
- Mechanism: `SpecDecodeBaseProposer.propose()` runs a plain Python `for token_index in range(self.num_speculative_tokens - 1)`; step 0 comes from a one-shot forward outside the loop. Host Python drives every draft step.
- Termination: host, fixed k; no data-dependent termination inside the loop.
- Token feedback: in-place device buffer writes per step: `self.input_ids[:batch_size] = input_ids`, `self.hidden_states[:batch_size] = hidden_states`; `_update_positions_dependent_metadata` bumps `max_seq_len += 1` and invalidates cached layout. int32 casts are "crucial when eagle model is compiled" (torch.compile path).
- Graph boundaries: PIECEWISE per-step graphs via `CudagraphDispatcher`; capture over the same persistent buffers in `dummy_run`. The loop is NOT one graph — host replays k per-step graphs. EAGLE CUDA-graph support landed ~v0.9.1.
- Measured overheads: no per-step ms published. Aggregate: "up to 2.5X" throughput, 1.8X request latency (8B, Red Hat blog); acceptance-rate draft metrics added alongside.
- Sources: https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py ; https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding ; https://docs.vllm.ai/en/latest/features/speculative_decoding/
- Confidence: high for loop/graph structure (source read); medium for perf claims (vendor blog, no per-step numbers).

## SGLang EAGLE
- Mechanism: `EAGLEDraftCudaGraphRunner.capture_one_shape` hands `run_once = eagle_worker.draft_forward(forward_batch)` to the backend; `draft_forward` loops `speculative_num_steps` internally — ALL steps baked into ONE CUDA graph per batch size. A megakernel path (`megakernel.run_draft_loop`) serves selected batch sizes with the draft loop fully fused into one launch. Defaults: steps 5 (Llama/Grok) / 3 others; draft tokens 8 / 4.
- Termination: host, per round, after verification. The graph "always runs the full speculative_num_steps"; no device-side early exit. All-greedy batches replay a separate `_GREEDY_VARIANT` graph that "skips the draft proposal distribution" — branching is a capture-time variant choice, not device control flow.
- Token feedback: persistent device buffers: `input_ids` (bs x width), `topk_p`, `topk_index`, recurrent `hidden_states`, and `out_cache_loc` sized `max_num_token * speculative_num_steps` — KV slots for ALL steps allocated upfront. Replay: grouped foreach copy of topk_p/topk_index, `copy_` for hidden_states, pad to bucket, replay under `device_timer_ctx("eagle_draft")`.
- Graph boundaries: one graph per (batch size, sampling variant); output 4-tuple (parent_list, top_scores_index, draft_tokens, draft_probs) sliced to real bs post-replay.
- Measured overheads: no ms numbers. EAGLE-3 373.25 tok/s vs 158.34 baseline (8B, H100). Host-sync cost evidence: publishing `seq_lens_cpu` per block "is a device-to-host sync per block" that "dominates the throughput of a 16-slot block" (LiLiCorr note); issue #23500 flags needless flashinfer syncs.
- Sources: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py ; https://docs.sglang.io/advanced_features/speculative_decoding.html
- Confidence: high — source code read plus docs.

## llama.cpp
- Mechanism: host-driven per token. Each draft impl builds a host batch, calls `llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch)`, and samples on host: `common_sampler_sample(smpl, ctx_dft, i_batch, true)` reads logits back — at least one backend round trip per drafted token. Impls: draft-simple, EAGLE3 (prenorm hidden state fed back via `batch.set_embd`), MTP (`llama_get_embeddings_nextn_ith`), DFlash2, DSpark, ngram.
- Termination: host, per token — a confidence gate on top-1 prob: `if (cur_p->data[0].p < params.p_min) drafting[seq_id] = false`, plus n_max / n_min caps. This is the only true early-exit-on-confidence found anywhere, and it pays a ping-pong per token.
- Token feedback: host arrays — drafted token re-added to `common_batch` at the next position for the next decode; optional backend sampling offload (`llama_set_sampler`) but logits still return per token.
- Graph boundaries: none — no CUDA graphs anywhere in the stack; one decode call per token.
- Measured overheads: none published per step; community reports net slowdown on Metal (-11% to -24%).
- Sources: https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp ; https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Confidence: high — source read; repo evolves fast, pin by commit before quoting.

## HuggingFace transformers
- Mechanism: the draft chain is a full `assistant_model.generate()` host call per round inside `_assisted_decoding`; KV `crop(-tokens_to_remove)` after verification.
- Termination: host. `_calculate_new_tokens` = `min(int(self.num_assistant_tokens), remaining)`; heuristic scheduler +2 on full match / -1 floor 1; `assistant_confidence_threshold` stops the assistant early (host-side stopping criteria).
- Token feedback: host — returned sequences plus stashed `past_key_values`.
- Graph boundaries: none; no CUDA graphs in `candidate_generator.py`.
- Measured overheads: none in-file.
- Source: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/candidate_generator.py
- Confidence: high — source read; included for the dynamic-depth scheduler pattern, not as a CUDA-graph reference.

## What this changes on the row
- Precedent confirmed: SGLang (whole `draft_forward` in one graph) and TRT-LLM one-model ("entire drafting loop as a single CUDA graph") validate merging round+steps into ONE graph. vLLM is the laggard — host-replayed per-step PIECEWISE graphs; copy the SGLang shape, not vLLM's.
- Nobody reads the drafted token on host between steps; all inter-step feedback rides device buffers (`input_ids`, `topk_p/topk_index`, `hidden_states`). Your current mapped-memory read per drafted token has no analogue in any leading engine.
- Device-side early exit on min_p gates: no implementation does it inside a graph. Consensus is ungated fixed depth + post-hoc trim (TRT-LLM KV rewind "blocks just marked as available"; HF crops `past_key_values`). Your "ungated + post-hoc trim" plan is the industry-standard shape; device-side gating would be novel with zero supporting precedent.
- min_p / confidence early exit exists only host-side (llama.cpp `p_min`, HF `assistant_confidence_threshold`) and costs a sync per token — evidence that your per-token host read is the anti-pattern, not the ungated chain.
- Staged Philox inputs: no direct evidence found. Closest pattern: SGLang capture-time variant graphs (greedy graph skips the proposal distribution) — RNG handling resolves at capture time, not via device branching. Verify Philox offset staging against SGLang's rejection-sampling capture before claiming parity.
- Pre-allocate per-step resources before launch: SGLang sizes `out_cache_loc` for all steps upfront and pads to capture buckets — mirror that with your staging buffer plus per-step KV slots so the merged graph never allocates mid-flight.
