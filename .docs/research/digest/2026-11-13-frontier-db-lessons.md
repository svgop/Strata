# Frontier database/storage-science audit (2026-11-13) — buffer pools/LSM/checkpointing vs our caches and tiers

Batch 3 of the 18-agent frontier wave. Workload frame: warm 97.6% hits — every positive gain is a CHURN-phase
claim. Local priors honored: #991 (admission-only loses at 2-8k slots) and R4.2g (per-layer quotas necessary).

## Ranked findings (positive ports, in order)
1. **KV residency: plain CLOCK -> PostgreSQL usage_count** (kv_stream.cu:116-133 binary ref bit; PG's 5-bit
   saturating counter survives hand passes — a block touched K times survives K passes; same GPU cost).
   TRIVIAL port, best cost/benefit of the digest. The parallel block-scan hand/epoch pins/bounded sweep are
   already better than textbook (partial negative).
2. **TinyLFU/W-TinyLFU admission filtering at the VRAM tier** — admit() never filters (expert_cache.cpp:
   643-661); a one-hit expert takes a scarce slot like a hot one, then pays a PCIe blob swap at rebalance.
   The DB claim: in scarce caches ADMISSION is worth more than eviction. Extends (not contradicts) #991 —
   that measured admission-ONLY; the filter+eviction shape is the third point on the curve. Per-layer 4-bit
   Count-Min + halving; rides the #991 A/B harness; medium gain on churn only.
3. **bgwriter-style adaptive rebalance cap** (fixed rebalance(8)/kRebalanceEvery -> sized from last window's
   measured n_miss; the between-forwards invariant already correct). Trivial.
4. **STMM-style per-layer quota retuning** (static equal quotas expert_cache.cpp:627-636 -> boundaries move
   a few slots/window from measured per-layer miss cost; hy_cpu/hy_gpu counters already collected). Low.
5. **LIRS/ARC IRR blend into the score-decay victim** — burst pollution (a prompt phase touches every
   layer's experts once) is the known failure of decayed-LFU; bounded gain (decay already forgets within
   log2(window) rebalances); the DS4 kLru policy + #991 harness exist. Medium; measure before adopting.
6. **Prefetch discipline: the dormant RouterLookahead** — queue discipline already matches DB practice
   (demand priority, stale-cancel, waste accounting = SEQDETECT's feedback loop); the gap is feeding the
   router-derived layer-ahead signal (stronger than any sequence detector — the gate output IS the access
   pattern) into the tier above the page cache. Small-medium cold/churn.

## Explicit negatives (already best practice — do not touch)
Direct-I/O buffer ownership = Stonebraker 1981 doctrine with 2020s mechanics (the RAM-floor gate = Oracle
shared_buffers sizing discipline); conversation cache = fuzzy checkpoints (ARIES no-force/steal pin
semantics; refusal-over-eviction) + dependency invalidation (materialized-view pattern; drop_superseded's
containment) — LRU correct here (strict recency access, 97-99% reuse); dense indexing beats cuckoo/robin-
hood for dense small-integer keys (expert ids, block ids) — any hash is strictly worse; tier migration
already LSM-shaped (amortized off-demand down-tier = compaction's invariant; the amplification arithmetic
answered correctly for 97.6% warm); score-decay greedy IS the IRM-optimal policy with a forgetting factor
(Mattson 1970/Belady lineage) — findings 1-2 are its known failure modes, not a family change.

## Bottom line
The engine independently reinvented much of the post-2005 DB state of the art. Four positive ports:
usage-count CLOCK (trivial), adaptive rebalance cap (trivial), STMM quotas (low), TinyLFU admission (medium,
the only one needing care).

Sources: ARC FAST'03; LIRS SIGMETRICS'02; LRU-K SIGMOD'93; TinyLFU TOS'17; Caffeine wiki; CLOCK-Pro ATC'05;
Mattson 1970; PG bgwriter + clock-sweep; LSM 1996; RocksDB; FAST'22 amplification; pB+-tree SIGMOD'01; DB2
SEQDETECT; TIP; database cracking; STMM VLDB'06; HyPer ICDE'11; ARIES TODS'92; materialized-views survey;
Stonebraker CACM'81; Oracle buffer pools; pg_prewarm; cuckoo/robin-hood papers.
