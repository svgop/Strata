# Frontier long-context serving audit (2026-11-13) — radix/HiCache/Sarathi/CacheBlend vs our 262K stack

Batch 2 of the 18-agent frontier wave. Scope correction established: our prefill chunk is NOT fixed 4096 —
`--prefill auto` bisects a 256-token grid under a byte-budget ring then host-prices a plan modeling
page-cache/PCIe/drive class (generate.cpp:7835-7990) — AHEAD of vLLM's static budget.

## Ranked findings
1. **Cross-conversation automatic prefix reuse (radix class) — the structural gap.** Reuse is exact-prefix,
   slot-anchored, O(entries x tokens) scan (conversation_cache.hpp:234-247,314-330); cross-conversation only
   via manual pin. Two conversations sharing a 5-30K system head each pay full prefill. SGLang radix 6.4x on
   agentic; CacheBlend: ~77% of RAG KV blocks only hit via non-prefix matching. 80% of the port exists
   (checkpoints are prefix snapshots; borrows() mounts across entries; host KV images identity-layout) —
   missing: block-hash index over parked ids + auto-pin + read-only mount. MED port. The #1503 leak class is
   the risk surface; take_reuse_for's identity gate is the pattern.
2. **Hot KV disk tier during serving** — disk appears only in T66 exit-drain; eviction drops KV to nothing
   (conversation_cache.hpp:418-428). vLLM tiered offload >2x throughput past ~128 conversations; the T66
   numbers prove disk read ~10x cheaper than re-read. Park-to-disk policy + lookup index; advisory tier.
3. **Decode KV-miss resolve sits ON the critical path** — 11.6% miss at 262K, synchronous zero-copy reads,
   no overlap; AND the proven KV_PREFETCH (+5.3% @32K prefill, docs/KV_PREFETCH.md) is OFF BY DEFAULT
   (built-but-unarmed, again). Plus an information advantage no frontier engine has: the indexer scores
   predict next round's blocks → speculative prefetch input. LOW-MED port.
4. **Decode-interleave granularity** — yield cadence is pieces of STRATA_BATCH_YIELD_CHUNKS=8 chunks (~32K
   tokens / ~4.6s between decode steps under a long prefill); BYIELD is reactive. Sarathi budget-mixing
   = the shape; a proactive yield (any decoding slot → yield at boundary) is a small condition change at
   give_way. LOW port, needs the ABAB.
5. Non-prefix KV reuse (CacheBlend ~3x on RAG fraction) — HIGH port (position-embedding correction,
   QSA re-scoring); KL-gated.
6. KV byte cuts (4-bit K per KIVI / block-run codec) — the "structure-aware coding owed" debt; K8V4's
   exact-score INT8-K is a DOCUMENTED deliberate decision; quality floors gate. MED-HIGH risk.
7. Prefix-aware VRAM residency (clock sweep is prefix-blind) — hygiene prerequisite for 1; pinned-bit per
   block in the spare ctl ints.
8. DCP sequence-sharding — boundary, don't build on PCIe consumer fabric (law 15).

## Already matching/leading
T66 persistence AHEAD (llama.cpp manual-hook only; HCache 5.73x TTFT research = our restore already ~10x
recompute); chunked-prefill planner AHEAD of static budgets; BYIELD = cooperative preemption that keeps work
(vs vLLM recompute); zero-copy KV L2 = HiCache mechanism; QSA = the DSA/NSA class run in-house (dense
fallback costs 17-22% accuracy — token-eviction research targets dense engines, not us); KV quant beyond
FP8-class (int8/Q4_0/K8V4 816-1056 B/cell, exact-score K); prewarm ahead (no frontier engine ships boot-time
expert prewarm); elastic arenas match/lead (25-30x vs re-read).

Sources: SGLang/RadixAttention arXiv:2312.07104; Sarathi-Serve arXiv:2403.02310; vLLM tiered-KV blog;
SGLang HiCache docs; DeepSeek DSA; vLLM FP8-KV blog; SnapKV; Quest; CacheBlend; HCache; Mooncake; vLLM DCP
docs; NSA; cuDNN NSA kernels; llama.cpp #20572; hermes-agent #35575.
