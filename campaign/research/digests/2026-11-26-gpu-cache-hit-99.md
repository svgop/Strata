# GPU cache hit rate → 99%+: the low-level method map (2026-11-26)

Operator target: VRAM expert-cache hit 98.4% → 99%+ ("higher is more t/s"). Confirmed by the
window anatomy: the miss classes are priced terms — CPU-expert 2.5-4.4% of routed entries =
2.9-4.1 ms of every ~14 ms window (the PCIe share, 0.4-1.2%, is the SMALL class). Cutting misses
~2/3 ≈ +10-15% on the affected windows. All numbers below are from tonight's logs and the tree
at HEAD unless cited.

## 1. What the tree already owns (law 5 — read what exists)

| organ | site | state on the standing |
|---|---|---|
| Frequency-adaptive tier | `PeerExperts::adapt` (peer_experts.cpp:335): swap-in RAM experts with usage ≥ 2.0 over cached experts, hysteresis margin +1.5, overlapped H2D on `refill_` stream, evicted victim → CPU compute meanwhile | LIVE (the "139 rounds, 5996 experts swapped in" lines) |
| **Usage counter** | `d.usage += 1.0f` per routed pair (expert_source.cpp:5272) — **no decay, no reset** | LIVE — the long-run-uniformity failure mode, both papers name it |
| Router look-ahead | `d.lookahead->submit(layers, x_f, n_tok, d.host_res)` — "CS-T: warm layer + 1": next layer's routing predicted from current hidden states | LIVE (record :3647-3652) |
| ForesightSwap | expert_source.cpp:6323-6560: per-layer GPU swap space, own DMA streams, miss-count admission ("admit after N misses, reuse after D windows"), predicted copies | BUILT, GATED OFF (`STRATA_FS_SLOTS` unset) |
| Route tail-skip | `STRATA_ROUTE_TAIL_SKIP` default 7 — rank ≥ 7 misses neither copied nor computed | LIVE, lossy by design (not a lever) |

## 2. Banked refutations — the boundary this queue must not cross (law 8/12)

F-125/126/127 closed the **"buy more residency" family** with numbers: residency-biased routing
(F-125: 7.11→7.88 worse), DMA-share increase (F-126: CPU −0.96 ms but GPU wait +6.03 ms),
slot capacity (F-127: +998 slots no effect, 6-conc −23%). The CPU term is STRUCTURAL (verify
blocks on routed-to-CPU experts), not capacity-limited. Any external method whose mechanism is
"more/bigger cache" is dead on arrival. The record names the one untested structural lever:
**overlap the CPU tier with the GPU kernel** (F-125 arithmetic: removing the term = 276 t/s class
on that workload's windows).

## 3. External intel — production-proven vs paper-only

**Production-proven.**
- **MoE-Infinity** (OSDI-class system, open source, single-consumer-GPU target = our exact class;
  3.1-16.7× TPOT vs vLLM/DeepSpeed/Ollama on PCIe4): request-level **EAM tracing** — the L×E
  activation matrix accumulated per request; a small **EAMC** (≈3% of recent requests, hundreds of
  entries) holds the 10-30 recurring activation groups; a new request matches its rEAM against the
  EAMC (cosine, 21 µs/query at 1K entries) and the matched group's matrix drives BOTH prefetch
  (place this request's hot experts before decode hits them) and replacement. Core finding: reuse
  is skewed WITHIN a request, uniform ACROSS requests — so global counters go flat, and layer+1
  dependency predictors over-fetch. ~30-50 requests to adapt after a workload shift. One pinned-DMA
  I/O thread saturates PCIe 4.0. https://arxiv.org/html/2401.14361v3
- **Ours already validates the hard part**: the tree's tier is frequency-based (not LRU), the
  look-ahead is live, evictions overlap with copies. The tree is past the naive baselines the
  papers beat — the remaining gap is the REQUEST-LEVEL signal.

**Paper-only (measured but not production).**
- **CMU caching/pre-fetch analysis** (arXiv 2511.05814, Nov 2025; Mixtral 8x7B on A100/A6000/L40/3090,
  bs=1): LFU beats LRU decisively (84.6% t/s on A6000); **caveat named: pure LFU lets popular experts
  go unevictable — combine popularity with unused-count (aging)**; speculative layer+1 loading hits
  84.6% precision/recall; **un-overlapped prefetch is NET NEGATIVE** (wrong guesses add transfers
  that compete with demand loads). https://arxiv.org/pdf/2511.05814
- **fMoE** (arXiv 2502.05370): per-expert priority prefetch/evict, expert maps known before the
  layer runs. https://arxiv.org/html/2502.05370v1
- **SpecPrefetch** (OpenReview, under review): tiny per-layer adapters predict next-layer priorities,
  router frozen. https://openreview.net/forum?id=lju1ZQpaII

## 4. The priced queue for 99%+ (law 15 — needle before the run)

1. **AGE THE USAGE COUNTER** (cheapest, triple-sourced, one-line class): decay `d.usage` per adapt
   round (env-gated, e.g. `STRATA_TIER_DECAY=0.90`) or halve at request boundaries. Removes the
   long-run-uniformity failure both papers name; the +1.5 margin then compares CURRENT-horizon
   frequencies. Needle: if the flat-counter noise contributes even a third of tonight's 2.5-4.4%
   CPU-miss share, the term moves ~1-1.5 ms/window. Arm: ABAB on the anatomy's CPU/PCIe shares +
   M1/conc, M3b watched (routing-adjacent change → the gate runs).
2. **REQUEST-LEVEL EAMC ADMISSION** (the structural candidate, production-proven externally):
   capture the per-request EAM at prefill (the routing already touches every pair — the counter
   exists), match against a small EAMC, and on match pre-place the request's predicted hot experts
   through the existing `refill_` stream DURING prefill (copy engines idle while SMs work — the
   overlap the papers demand is already the tree's shape). Needle: the CPU-miss share cut toward
   ~1/3 = +10-15% conc windows; M1 decode benefits via fewer mid-window swaps. Cost: L×E floats per
   request + a few hundred cached matrices + 21 µs cosine — noise against 14 ms windows. Risks:
   workload shifts (gates vs chat) cost ~30-50 requests of adaptation (measured externally);
   prefetch transfer volume must stay overlapped or it nets negative (CMU's warning).
3. **HYSTERESIS SWEEP** (cheap arms on live policy): the 2.0 admit threshold, +1.5 margin, and
   max_swaps are tuning surfaces F-127 never touched (F-127 closed capacity, not policy). A 3-point
   sweep pricing the CPU-term vs swap-churn tradeoff; the anatomy lines measure it directly.
4. **FORESIGHTSWAP ARM** (built, gated, never armed on this lane): `STRATA_FS_SLOTS` with its
   miss-count admission is a hybrid demand/predictive swap space — upstream measured the VRAM-slot
   form negative, but this form (miss-admission + reuse window + own streams) differs. One scratch
   boot prices it.
5. **DRAFT-DRIVEN PREFETCH** (tree-native, novel): the MTP draft already runs ~1 window ahead of
   verify; its routing decisions preview the verify's expert needs at WINDOW depth (beyond the
   layer+1 look-ahead). Submit the draft's accepted routes as prefetch hints. Paper-free — the
   predictor already exists in-tree. Price after 1-2 land: the same CPU-term arithmetic.
6. **CPU-TIER OVERLAP** (the record's own untested structural lever, biggest prize): compute a
   layer's CPU experts on the host pool while that layer's GPU kernel runs, instead of verify
   blocking (F-125: removing the term = 276 t/s class). Larger engineering; instrument first.

**Not candidates** (refuted, law 8): more slots, bigger cache, residency-biased routing, DMA-share
bumping, reserve tuning.

## 5. Feasibility statement

98.4 → 99+ means removing 40-60% of the miss classes. The miss classes are the CPU-expert term
(2.5-4.4% of entries, 2.9-4.1 ms/window) plus PCIe (0.4-1.2%). Levers 1-3 address retention
quality, lever 2 addresses prediction, lever 6 removes the term's cost even at constant hit rate.
MoE-Infinity's production numbers (near-full-GPU latency from a 24 GB card over PCIe4) put the
ceiling within reach on this hardware class. Order: 1 → 3 → 4 (cheap arms, one battery), then
2 (the real candidate), then 6.
