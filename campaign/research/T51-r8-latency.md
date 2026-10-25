# T51 / R8 — Tail-Latency & Responsiveness in LLM Serving (dossier)
Date: 2026-10-25. Researcher: R8. Scope: tail control for OUR rig (1x RTX 5090, Qwen3.8-Flash-Next MoE, MTP spec-dec window T=6, 6 decode slots, 2 cohorts).

## Our measured baseline (the targets)

| Metric | Now | Goal |
|---|---|---|
| TTFT p50 / p95 / p99 | 0.3 / 0.9 / 2.0 s | p99 < ~1.2 s |
| ITL p50 / p95 / p99 (6-conc) | 4.5 / 18.8 / 19.8 ms | p99 < ~10 ms |
| ms/window drift in a probe set | 14 ms early → 10 ms late | — (early tail = interference/prefill pollution) |
| Admission stall between rounds | 30–56 ms (sync'd) | minimize |
| Slot copies (materialize/demote) | 40–90 ms | minimize/hide |
| Parking restore | 9–15 ms | hide |
| Long-prompt prefill | 88k tok = 12.5 s, chunked at 32k, blocks decode lane | never block decode lane for seconds |

Throughput records that must survive: solo ~220 t/s, 6-conc ~306 t/s.

---

## PART 1 — PER-SOURCE FINDINGS (graded A–D for relevance to OUR rig)

### P1. Sarathi-Serve: "Taming Throughput-Latency Tradeoff in LLM Inference" (OSDI'24, Agrawal et al.) — GRADE A
- Sources: https://www.usenix.org/conference/osdi24/presentation/agrawal , https://arxiv.org/html/2403.02310v1 , https://www.usenix.org/system/files/osdi24-agrawal.pdf
- Mechanism: chunked prefill (split prefill into near-equal chunks) + stall-free batching (admit new requests into a running hybrid batch without pausing decodes). Per-iteration token budget, profiled once: pack ALL running decodes first, then one prefill chunk, then admit new requests only in leftover budget. Decode iterations always get priority.
- Quantitative: naive hybrid batching (decode + full prefill) inflates TBT up to **28.3x** vs decode-only. Yi-34B, budget 1024 (Table 4): hybrid-only p50 TTFT 0.56 s / p99 TBT 1.08 s; chunked-only 1.04 / 0.34; combined 0.85 / **0.29**. Stall-free batching alone is ~half the latency gain (chunking removes the 28x spike; stall-free removes the admission pause).
- Budget/chunk-size practice: budget **512 for strict TBT SLO, 2048 relaxed**, 1536 for 70B. Smaller chunks add overhead: attention re-reads prior KV + allreduce fixed cost; linear-op overhead ~zero at chunk 2048. Tile quantization: chunk sizes must be powers of two — 257 vs 256 costs **+32% prefill time**.
- Why piggyback is nearly free: decode is memory-bound; batch exec time rises little until ~500–600 tokens, so a small prefill chunk rides a decode window essentially for free (weight fetch amortized). This is the theoretical basis of OUR short-read-64 riding.
- TTFT cost of chunking: chunked-only raises the chunked request's own TTFT (1.04 vs 0.56 s in Table 4) — tunable via chunk size. Budget is a fixed per-deployment knob in the paper; dynamic budgets left as future work.
- Fit to us: we already ride prefill on decode windows (short-read 64) — but we have no *token-budget* discipline on the ride, and admissions are synchronous (the exact "stall" Sarathi's stall-free batching removes).

### P2. DistServe (OSDI'24, Zhong et al.) + Splitwise (ISCA'24) — GRADE B for a single GPU
- Sources: https://arxiv.org/html/2401.09670v2 , https://haoailab.com/blogs/distserve , https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin , https://gimletlabs.ai/blog/multivendor-prefill-decode-disaggregation
- Mechanism: separate prefill and decode onto different GPU replicas; per-replica parallelism tuned independently; **goodput** = fraction of requests meeting BOTH a TTFT SLO and a TPOT/ITL SLO (joint metric, not averages). Up to 7.4x goodput vs collocated vLLM; Splitwise reports 2–7x via phase-specialized hardware.
- 2026 unifying view (https://www.alphaxiv.org/abs/2508.01989): collocated/chunked systems win TPOT-bound SLO regimes; disaggregated systems win TTFT-bound regimes.
- Fit to us: physical disaggregation impossible (one GPU). The transferable parts: (a) measure goodput (% of probes meeting TTFT + ITL SLOs simultaneously) alongside p50/p95/p99; (b) schedule TTFT-critical work and ITL-critical work as separate *logical* lanes even on one GPU (time-sliced separation = what our cohorts already approximate).

### P3. AdaServe: SLO-customized serving via fine-grained speculative decoding (EuroSys'26, Li et al.) — GRADE A-
- Sources: https://arxiv.org/html/2501.12162v2 , https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf
- Mechanism: per-request **speculation length as the latency/throughput knob** + admission control that only admits requests whose SLO can still be met. Each request is decoded at the speed its SLO demands; spare capacity converts to throughput.
- Quantitative: SLO violations down up to **4.3x**, goodput up **1.9x** vs best baselines; +73% SLO attainment.
- Fit to us: we run a FIXED MTP window T=6 for everyone. AdaServe is the direct precedent for variable per-cohort window T: interactive cohort shorter T (less wasted verify on acceptance collapse → tighter ITL tail), batch cohort T=6+ (throughput). This is the only paper-grade evidence that T is the right ITL-tail knob.

### P4. FlowPrefill: decoupling preemption from prefill scheduling (arXiv 2602.16603, Feb 2026) — GRADE A-
- Source: https://arxiv.org/html/2602.16603v1
- Mechanism: preemption at OPERATOR boundaries (qkv_proj/attn/o_proj/gate_up/down_proj) inside a prefill chunk — cooperative suspend via signal/ACK; event-driven scheduling (2 rounds/request) with slack-aware EDF.
- Quantitative: preemption latency bounded by one operator — **3.5–4.2x lower blocking than layer-level, all < 4.5 ms**; 4.7–5.6x sustainable rate vs DistServe; ~2.0x vs DistServe+chunked-prefill-2048; 1.6x better TBT SLO attainment than vLLM-CP2K; 1.6x goodput on Qwen3-30B-A3B MoE.
- Chunk-size guidance (matches Sarathi): ~2K tokens too fine (split overhead), ~16K too coarse (granularity), intermediate best; budgets 4K and 8K similar, larger budgets raise SLO-violation risk.
- Fit to us: the 88k-prompt 12.5 s decode-lane block is exactly the problem FlowPrefill attacks. Cheapest transfer is not full operator-level preemption but chunk-size reduction + guaranteed decode-window interleave between chunks (Sarathi-style), with FlowPrefill as the upper bound of the design space.

### P5. KVFlow: workflow-aware eviction + prefetch on SGLang radix cache (arXiv 2507.07400, Jul 2025) — GRADE B+
- Source: https://arxiv.org/html/2507.07400v1 ; ecosystem: https://www.lmsys.org/blog/2026-08-11-unified-radix-cache , https://github.com/sgl-project/sglang/issues/27574
- Findings on agentic traffic: large fixed prompts reused every invocation; workloads are prefill-heavy; tool-call waits make LRU eviction wrong (LRU HiCache fell to 0.57x of GPU-only SGLang at 1024-token prompts/64 workflows). Fix: steps-to-execution priority written into the radix tree (suffixes evicted first), plus overlapped CPU→GPU prefetch so the next agent's KV is ready before it runs.
- Quantitative: up to 1.83–2.19x workflow speedup vs SGLang HiCache; gains grow with fixed-prompt length (1.48x avg at 8192 fixed tokens).
- Fit to us: our conversation parking/restore (park on demote, restore 9–15 ms on promote) is a two-slot version of this. Agentix (NSDI'26, https://www.usenix.org/conference/nsdi26/presentation/luo) reports >90% intra-program prefix-cache hit rates and 4–15x program throughput by scheduling LLM calls as *programs* with known future inputs — i.e., the next-to-return conversation after a tool call is predictable.

### P6. Speculative decoding vs ITL variance (production + papers) — GRADE B
- vLLM docs: spec dec reduces ITL at medium-low QPS; **at high QPS it is recommended OFF** for throughput — the draft/verify overhead converts to jitter under contention (https://docs.vllm.ai). Inworld's engineering writeup on spec-dec + chunked prefill makes the same point: draft steps block concurrent requests and window length sets the variance scale (https://inworld.ai). Reducto's analysis: accepted-token counts do not always translate to end-user wins (https://reducto.ai).
- Fit to us: with T=6, a window that accepts k tokens emits k tokens for one window cost. p50 ITL 4.5 ms ≈ k≈5 (window ~27 ms); p95 18.8 ms ≈ k=1 (window ~19 ms). The p99 tail is acceptance collapse + interference windows, so BOTH T and per-window interference must move to get p99 < 10 ms.

### Minor/secondary
- **Nexus** (arXiv 2507.06608): intra-GPU prefill/decode disaggregation, eliminates interference via spatial partitioning — multi-GPU orientation, concept only. GRADE C+.
- **TetriInfer** (arXiv 2310.12134): predicted-ITL pipeline execution, opacity-aware matrix splitting for contention. GRADE C.
- **SLOs-Serve** (arXiv 2504.08784): multi-SLO chunked scheduling, good taxonomy of TTFT vs TBT SLO classes. GRADE C+.
- **Prefill-decode interleaving goodput** (arXiv 2504.14489): caps token budget = prefill chunk + decode batch to guarantee decode SLO — confirms budget arithmetic. GRADE C+.

### Production sources
- **S1. vLLM V1 scheduler (docs + code)** — GRADE A. Chunked prefill ON by default; online-serving default `max_num_batched_tokens=2048` (offline LLM class 8192); with chunked prefill the policy is **decode-first** (decodes scheduled into the budget first, leftover to prefill chunks); docs: lower the budget (2048–8192) for ITL, raise >8192 for throughput; async scheduling overlaps CPU scheduling/prep with GPU work, known gap interactions with preemption remain. Sources: https://docs.vllm.ai/en/stable/configuration/optimization , https://docs.vllm.ai/en/latest/api/vllm/config/scheduler , https://github.com/vllm-project/vllm/issues/18547 , https://github.com/vllm-project/vllm/issues/27679 , https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1
- **S2. TensorRT-LLM CapacityScheduler** — GRADE A-. Three policies: `GUARANTEED_NO_EVICT` (default; admit only if the WHOLE sequence fits KV — no preemption, predictable tails), `MAX_UTILIZATION` (throughput, evicts → tail risk), `STATIC_BATCH`. The admission-before-commit pattern is exactly what our between-round admission should mirror. Source: TensorRT-LLM `useful-runtime-flags.md` (github.com/NVIDIA/TensorRT-LLM).
- **S3. NVIDIA Dynamo "Agents on SGLang" guide** — GRADE B. Prefix-heavy agent traffic makes KV reuse the TTFT lever; covers prefill-chunk scheduling under agentic loads. https://docs.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/agents-on-sg-lang
- **S4. SGLang KV-hints RFC #27574 + unified radix cache (Aug 2026)** — GRADE B. Applications may pass KV-cache hints, but the SCHEDULER retains ownership and may clip/defer/reject — the right ownership split for our parking hints too. https://github.com/sgl-project/sglang/issues/27574
- **S5. vLLM spec-dec docs / Inworld / BentoML practice** — GRADE B-. See P6.

---

## PART 2 — PROPOSALS (costed against OUR numbers, ranked by p99-per-effort)

### P-A (rank 1). Decode-lane-safe short-read budget + decode-first window packing
- **Mechanism / which tail:** ITL p99 19.8 ms and the early-window 14 ms interference tail. Bound per-window riding prefill: `sum(riding_prefill_tokens + decode_window_tokens) <= B` with decodes packed first (Sarathi budget; vLLM V1's 2048 decode-first policy in miniature). Our probe delta (early 14 ms vs late 10 ms) prices the interference at ~4 ms/window average with spikes to ~10 ms — riding chunks are the only per-window variable work at 6-conc.
- **Derivation of the cut:** decode-only windows run 10 ms. Removing riding-prefill spikes from the p99 region: p95 18.8 → ~12–14 ms; p99 19.8 → **~12–14 ms** expected after this alone (remainder is spec-window acceptance variance, see P-D).
- **Cost:** LOW. A budget check at the window-composition site + a probe run. Riding at short-read 64 contributes little measured throughput at 6-conc (Sarathi: free ride needs decode batch headroom; at 6 slots we have it, but the budget must still cap it).
- **Risk:** solo/short-prompt throughput dips if the budget is set too tight — keep B adaptive: full 64 when no decode cohort active, ~16–32 tokens when it is.
- **Cheapest falsifier:** one 6-conc probe with short-read riding disabled entirely: if ITL p99 does not drop below ~13 ms, interference is not the tail driver and this proposal is dead.
- **Port site:** `/home/svg/LLM/strata-custom-2/src/program/generate.cpp` window-composition/short-read site (the prompt-rides-decode path); budget constant surfaced like `PP_CHUNK_MAX` in `/home/svg/LLM/strata-custom-2/serve/server.py` (line ~287).

### P-B (rank 2). Interleaved bounded prefill for long prompts (stall-free chunks; FlowPrefill-lite)
- **Mechanism / which tail:** TTFT p99 2.0 s and multi-second decode freezes from 88k-prompt prefills (12.5 s at 32k chunks). Replace "prefill phase hogs lane for seconds" with: prefill_chunk reduced to 4–8k AND one decode window forced between consecutive prefill chunks (guaranteed decode cadence, Sarathi stall-free in reverse).
- **Derivation of the cut:** 88k in 12.5 s ≈ 7.0 kt/s ⇒ a 4096-token chunk ≈ 0.6 s; a decode window (10 ms) every chunk makes the worst decode gap ~0.6 s instead of ~4.6 s (32k chunk). TTFT p99 is dominated by requests queued behind/under these phases; capping the freeze to ~0.6 s pulls the p99 events (currently ~2.0 s tails) to ~0.9–1.2 s. MoE caveat from generate.cpp: bigger chunks stream fewer expert bytes per token, so expect the 88k prefill itself to slow ~10–25% (12.5 → ~14–16 s) — acceptable for a background leg, MUST be measured.
- **Cost:** MEDIUM. `prefill_chunk`, `prefill_auto_max` (8192 default; gfx12 16k ceiling), and `prefill_until` already exist in generate.cpp; the new piece is the interleave rule (yield lane after each chunk when a decode cohort is active). `PP_CHUNK_MAX=32768` in server.py is the admission-side mirror.
- **Risk:** MoE expert-stream amortization loss (above); power-of-two chunk sizes mandatory (Sarathi tile quantization, 257-vs-256 = +32%). Chunked request's own TTFT rises (Sarathi Table 4: 0.56→1.04 s) — the 88k requester pays, everyone else wins.
- **Cheapest falsifier:** mixed probe (one 88k prompt + 6-conc decode), chunk 4096 vs 32768: measure max decode-lane gap and short-prompt TTFT p99. If decode gap < 0.7 s and 88k prefill inflates > 40%, stop at 8192.
- **Port site:** `/home/svg/LLM/strata-custom-2/src/program/generate.cpp` prefill chunk loop (`prefill_chunk`/`prefill_auto`/`prefill_until`, lines ~693–719) + `/home/svg/LLM/strata-custom-2/serve/server.py` admission note at PP_CHUNK_MAX.

### P-C (rank 3). Async admission + pre-materialized slots (vLLM V1 overlap + TRT-LLM GUARANTEED_NO_EVICT admission test)
- **Mechanism / which tail:** admission stalls 30–56 ms and slot copies 40–90 ms; these land in TTFT (the `observe_admit` histogram in server.py line ~1795 records exactly this queueing) and in E2E/ITL tails whenever a round boundary coincides with a token emission. Move admission prep (control-line acquisition, ticket, slot planning) OFF the window loop: prepare the next admission during decode windows, commit it at a boundary; slot materialization goes through the existing background path (`EL_PROMOTE` regex already distinguishes "at the admission" vs "background"). Add the TRT-LLM-style capacity test: admit only if the conversation's projected KV fits the spare pool, so admission never triggers a mid-round demote cascade.
- **Derivation of the cut:** vLLM V1 attributes its latency gains largely to overlapping CPU control work with GPU execution; our 30–56 ms sync stalls + 40–90 ms copies overlap to ≤ 5–10 ms residual on the critical path. TTFT p99 benefit stacks with P-B (queueing behind long prefills is the other TTFT p99 term).
- **Cost:** MEDIUM-HIGH. Concurrency discipline around `ctl` lock / `ctl_ticket` FIFO (server.py lines ~1204–1210, ~1798–1807) is delicate; the F-14 ticket order must be preserved.
- **Risk:** control-line races; admission-order fairness regressions (mitigate: commit order stays FIFO, only prep overlaps). Capacity misprediction → park storms (mitigate: conservative fit test = TRT-LLM default).
- **Cheapest falsifier:** instrument-only run: histogram `observe_admit` and materialize times per round boundary; if admission stalls are already < 10 ms at p99 under agent traffic, skip the rework.
- **Port site:** `/home/svg/LLM/strata-custom-2/serve/server.py` admission path (`Service.admit`/`_admit_next`, latency sink wiring ~line 1002) + background materialize site (`strata elastic: slot materialized`).

### P-D (rank 4). Per-cohort adaptive MTP window T (AdaServe knob)
- **Mechanism / which tail:** ITL p99 via acceptance-collapse windows (p95 18.8 ms ≈ 1-token windows). Interactive cohort: T=4 (cheaper failed-verify windows, ~30% less draft+verify waste on collapse); batch cohort: T=6–8 (throughput preserved where acceptance is high). AdaServe showed per-request speculation length cuts SLO violations up to 4.3x at equal goodput.
- **Derivation of the cut:** collapse windows cost ~19 ms for 1 token today; at T=4 the same window ≈ 13–15 ms → ITL p95/p99 region moves to **~12–15 ms**; combined with P-A's interference cut, **< 10 ms** is reachable. Guard: only the interactive cohort's T changes; solo and 6-conc throughput records are batch-cohort legs and are untouched by construction (verify anyway).
- **Cost:** LOW as an experiment (T is a CLI/engine knob; `draft_policy.hpp` is the natural home for per-cohort policy), MEDIUM as a productized cohort-aware policy.
- **Risk:** lower T cuts solo ITL wins for the interactive cohort when acceptance IS high (p50 4.5 ms could worsen to ~6–7 ms); throughput of the interactive cohort drops if applied globally — do not.
- **Cheapest falsifier:** log per-window accepted-token histogram on the 6-conc probe; if the p99 ITL mass is NOT concentrated in k=1 windows, acceptance collapse is not the tail and T tuning is dead.
- **Port site:** `/home/svg/LLM/strata-custom-2/src/program/generate.cpp` `spec` window field (line ~727) + `strata/spec/draft_policy.hpp` cohort hook.

### P-E (rank 5). Predictive restore: pre-promote the tool-call successor (KVFlow/Agentix-lite)
- **Mechanism / which tail:** parking restore 9–15 ms sits on the TTFT of the first token after every tool call — the most frequent TTFT event in agent traffic. Agentix measured >90% predictability of the next LLM call in agent programs; KVFlow prefetches exactly this. On tool-call emission, start background restore of the conversation most likely to return (last-parked, same cohort) using the existing background materialize path; admission then finds it warm.
- **Derivation of the cut:** restores hidden under the 30–56 ms tool gap → ~0 ms on critical path; TTFT p50/p95 shave ~9–15 ms per multi-turn hop; small but free.
- **Cost:** LOW (background path exists — the `EL_PROMOTE "background"` branch).
- **Risk:** wasted bandwidth on wrong predictions (bounded: one speculative restore); VRAM pressure if both cohorts speculatively promoted (gate: only when spare chunks exist, TRT-LLM capacity-test style).
- **Cheapest falsifier:** count restore events whose conversation returned within 200 ms of its tool call — if < 60%, prediction is too weak.
- **Port site:** `/home/svg/LLM/strata-custom-2/serve/server.py` parking/restore event handling (`conversation cache: parked|restored`, line ~747–760) + admission.

### Ranking rationale (p99-per-effort)
1. **P-A** — tiny effort, directly attacks the measured 14 ms early-window interference; biggest ITL p99 ms per hour of work.
2. **P-B** — the only proposal that can move TTFT p99 2.0 → <1.2 s alone; medium effort because chunk knobs already exist; MoE amortization is the main cost to measure.
3. **P-C** — cuts the admission/copy tails (both metrics' E2E tails); highest engineering risk, do after P-A/P-B establish the interference picture.
4. **P-D** — cheap experiment, needed alongside P-A to get ITL p99 under 10 ms (interference removal alone leaves ~12–14 ms).
5. **P-E** — small but essentially free given existing background path.

### Goal accounting
- **ITL p99 < 10 ms** = P-A (interference tail: 19.8 → ~12–14 ms) + P-D (collapse windows: → ~12–15 ms, compounding) + P-C (between-round stalls out of the ITL/E2E lane). Throughput guard: T unchanged for batch cohort; P-A budget adaptive.
- **TTFT p99 < 1.2 s** = P-B (long-prompt freezes capped ~0.6 s) + P-C (admission wait off critical path) + P-E (restore hidden). The 2.0 s p99 decomposes as queueing behind multi-second prefill phases + sync admission; both terms are addressed above the 1.2 s line with margin for the chunked requester's own TTFT increase.
- **Throughput records**: only P-B's chunk-size change touches prefill efficiency; its falsifier quantifies it first. Everything else rides existing headroom (decode windows are memory-bound; Sarathi's free-ride regime).

## Cross-cutting measurement note
Adopt DistServe's **goodput** framing as a campaign metric: % of probe requests meeting (TTFT < 1.2 s AND ITL p99 leg < 10 ms) simultaneously per leg — p50/p95/p99 alone can hide which tail a change actually moved.
