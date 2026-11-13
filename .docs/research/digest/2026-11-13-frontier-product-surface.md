# Frontier product-surface audit (2026-11-13) — vLLM/SGLang/llama-server API surface vs our serve layer

Batch 2 of the 18-agent frontier wave. Scope correction: our surface is BROADER than the brief assumed
(OpenAI chat + Anthropic /v1/messages + stateless Responses + /v1/models + Prometheus with vLLM's metric names
+ seed + logit_bias + reasoning tiers + drain + cancel propagation; DSML IS translated to standard
tool_calls). The real gaps:

## Ranked findings
1. **Structured outputs: NO constrained decoding** — prompt directive + validate-and-fail (structured.py:1-9;
   failure = HTTP 502). Frontier: xgrammar/llguidance default in vLLM/SGLang/TRT-LLM; adherence 61-72% → ~100%.
   M port (1-2 weeks) and the hook EXISTS: the host sampler applies per-request logit masks (verify.cpp:4666,
   logit_bias-proven plumbing); json_schema→GBNF is lineage code; compile-per-schema cache. Risks: spec-decode
   grammar-blindness; the DS CPU-lane stage placement needs a priced arm.
2. **logprobs/top_logprobs absent AND silently ignored** — the distribution is already materialized host-side
   at the sample point. M port; transport per token (the DSI1/GENI sidecar-file pattern is the precedent);
   mind the multi-byte detok edge.
3. **Legacy /v1/completions absent** — lm-eval-harness local-completions + a long tool tail hit it
   exclusively. S-M over the existing encode+run path.
4. Chat route 400s on structured+tools while the Responses route allows it — port the branch, hours
   (moot after 1).
5. stream_options.include_usage ignored (usage rides the finish chunk; spec wants an empty-choices final
   usage chunk — Datadog/LiteLLM branch on that shape). S.
6. /v1/embeddings + /v1/rerank absent (RAG stacks probe at setup) — M-L engine work (hidden-state readout +
   pooling); honest boundary option: name it, don't ship weak embeddings.
7. Batch API absent — S-M via the existing FIFO queue; useful for overnight eval batteries.
8. One port per model, no router mode (llama-server --models-dir LRU fleet) — M-L; cache identity must follow
   model switches or T66 cross-contaminates.
9. Silent-ignore of unsupported params (logprobs, n, stream_options, parallel_tool_calls) — warn-once or
   explicit 400 (S; warn-once first to avoid breaking "working" clients).
10. n>1 ignored (hardcoded index:0) — M; substrate exists (slots/replicas).
11. No TLS termination — S.
12. Audio: named boundary, both layers absent — record only.
13. Responses stateless (deliberate; /v1/responses/{id} 404) — S-M if clients demand it.
14. FIFO only, no priority/preemption — S-M priority field; true preemption is a graph-family event (scoping
    problem).

## Where we exceed the frontier
Position-keyed seed determinism (Philox(seed,pos0+t) — bit-exact through the spec stack; stronger than vLLM's
plain seed); per-request spec telemetry (draft_n/accepted) almost nobody surfaces; /v1/telemetry depth
(genuine per-token gap histograms, admission waits, stage topology) + vLLM-named Prometheus; lifecycle
(prewarm yields to traffic, drain 503+Retry-After, quiet-cancel proven, admission 429s, VRAM handback);
T66 conversation persistence across restarts (unique in family); MCP hub (stdio+HTTP, crash-restart —
llama-server is stdio-only, vLLM none); reasoning controls beyond parity; DSML translation with streaming
tool deltas + cut-off-call recovery + client-quirk armor (Claude Code billing pin, Codex keep-alive); vision
in-engine tower + elastic mmproj; security posture (constant-time keys, CORS guards — llama-server defaults
CORS to *).

## Bottom line
Two genuine week-one walls: grammar-class structured outputs (1) and logprobs (2) — both port into hooks that
already exist. The rest is HTTP-shape work ranked by which clients probe it first.

Sources: vLLM structured-outputs + online-serving docs; SGLang docs; llama-server README; SqueezeBits guided-
decoding study; GMI overview; XGrammar-2 (mlc blog); HF logprob-loss blog; OpenAI streaming-usage community
note; Datadog stream_options TIL; LangChain vLLM #7154; Cline #7079; llguidance buyer's guide.
