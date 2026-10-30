---
type: roadmap
id: strata-dsv41-roadmap
status: active
created: 2026-10-30
owner: campaign/probe-lane.py DS battery + this file's exit criteria per move
---

# dsv41 performance roadmap — first pass (D1–D10), 2026-10-30

Mission: lift the DS standing lanes (T56 bank: solo 52.3 / pair 51.6 / conc 57.5 t/s, --experts 5,
262K context) by attacking the round's sync structure, not its numerics. Source: four-investigator
code-only sweep of src/dsv41/ + the shared memory layer, 2026-10-30, all anchors spot-verified.

## MEASURED ANATOMY (T61, 2026-10-30 — supersedes every estimate in this file)

Same-hour 3-leg baseline (old binary): **51.5 / 49.3 / 56.2**. New binary arms-off parity:
50.1 / 50.3 / 55.9 (in-band). Round ≈ 19 ms/token decomposes as:
- **GPU layers 7.2–7.7 ms** (attention ~5.6–5.8 of it — the biggest GPU kernel class)
- **CPU lane 7.0–8.7 ms — MEASURED AT THE DRAM CEILING**: DS4_PROF per-phase breakdown
  (n=269K calls): entries/call 1.80, prep 0.008, **gate+up 0.241 ms ≈ 46 GB/s**, h-quant 0.005,
  **down 0.112 ms ≈ 50 GB/s**. ~36 expert GEMVs/token (~335 MB) at 46–50 GB/s of the ~50 GB/s
  box ceiling. The lane is byte-bound; only fewer bytes moves it. (My study-session estimate of
  "13.5 GB/s effective" was wrong: entries/call is 1.8, not 0.4.)
- **Ping-pong slack ~4 ms** (~100 µs/layer): the lane's 366 µs answer vs the GPU's ~180 µs of
  post-publish overlap work per pick-layer. The publish/overlap structure is already maximal
  (route_publish precedes the shared+VRAM GEMVs); the residual is the lane's answer being
  longer than the GPU's same-layer work. Structural, not fixable by reordering (see D2).
- **Head 0.29–0.30 ms** (D9's event-bracketed truth; head_ms's 9–10 ms was the DS-2 artifact —
  lane-wait residue draining into the final sync. Confirmed with a number.)

Config reality discovered: the CPU lane's experts are **copied into RAM at load** (110.4 GB,
"--mmap-experts auto: fit the RAM"), there is **no demand store** in the standing config, and
2,634 of 15,360 experts sit in VRAM (22.96 GB) covering ~82–92% of routed picks; dspark's
auto-off ("verify rounds read 3.3x a token's bytes and gain nothing") is CORRECT for this
lane-bound engine — its cost scales with verify width while its benefit doesn't.

## T61 verdicts

| Move | Verdict | Numbers |
|---|---|---|
| D9 telemetry | **DONE** | head_true_ms + DS4_PROF lane phases now live in the decode line |
| D1 single drain | measured, **below the +4% bar — arm stays env-gated** | 52.0 / 50.9 / 56.4 (2-leg) vs 51.5 / 49.3 / 56.2: +1.0 / +3.2 / +0.4 |
| D2 moe_down overlap | **FALSIFIED IN CODE** (win_experts: "the same bits, no faster") | — |
| D3 buffered store | **DEAD BY CONFIG** — no demand store engaged | — |
| D4 store memmove | **DEAD BY CONFIG** — same | — |
| D5 store lookahead | **DEAD BY CONFIG** — same (and naive arm already falsified in-code) | — |
| dspark in serve | **auto-off is correct** (lane bytes scale with verify width) | — |

## The remaining levers (re-priced by the measured anatomy)

1. **Fewer lane bytes** — the only lane lever that exists:
   - `--experts 4` (quality-traded, the T51 class): lane bytes −20%. Battery T61-4e below.
   - A lighter CPU-lane pack build (2.06 → ~1.6 bpw for the cold tail): a pack decision,
     quality-traded; the "already 1.75–2.31 bpw mixed" floor limits the headroom.
   - VRAM hit-rate tuning via the learned profile (online, no code; watch the 7–18% spread).
2. **The ~4 ms ping-pong slack** — the only structural "free" term: accumulate-as-entries-land
   partial combine (per-entry flags; the combine overlaps the lane's tail entries). Est. +3–5%.
   Medium kernel work; verify the combine's sum-order determinism story first.
3. **Attention (~5.6–5.8 ms/token)** — the biggest GPU class: nsys-guided kernel pass on the
   FP4-KV/indexer path (post-ATTN_SPLIT). No quick arm exists.
4. **Prefill** (946 t/s @8K first-light): its own lane; W19-class small-read work applies.

## Order of attack (current)

T61-4e battery → partial-combine design review (slack) → attention nsys pass → pack-level
byte decisions with the operator.

## The measurement-model correction (read first)

**The "head ~40% of round" belief is an accounting artifact.** In the standing config the engine
runs the async mailbox path (`sync_decode = getenv("DS4_SYNC") != nullptr`, ds4_model.cu:1025 —
unset in launch-custom-dsv41-8081.sh). On that path the per-layer lane waits happen on-device in
`wait_seq_kernel` (ds4_kernels.cu:1722-1730), so `st.gpu_wait_ms` (ds4_model.cu:9604-9606, the
host-routed path's counter) stays ~0, and every lane wait drains into `st.head_ms` at the final
`cudaStreamSynchronize` (ds4_model.cu:9058-9060). The true head (hc_pre_norm + 129,280-vocab GEMV
+ sampler) is a few hundred µs. Consequence: **do not optimize the head; cut lane latency and
overlap the lane.** Every move below is sized against that model.

Round anatomy (verified): `round()` (ds4_main.cpp:169-206) → `forward()` (ds4_model.cu:9011) →
per-layer mailbox: `route_publish` + `publish_kernel` → `serve_lane` service thread
(ds4_model.cu:6875-7038) computes CPU-lane experts while the GPU spins in `wait_seq` → head →
stream drain → `sample()` → `printf("T %d\n")` + `fflush` (ds4_main.cpp:883-886).

## Protocol (same law as the Qwen campaign)

1. Every change env-gated, default off, until adopted; A/B via env or binary snapshot
   (`cp engine/strata engine/strata.pre-d<n>`).
2. Gate = the DS lane battery (T52 pattern: solo/pair/conc, same-hour ABAB, 3v3).
   dsv41 outputs are boot-nondeterministic (T56 finding: A/A text diverges) — text A/B is
   invalid; chain-level isolated tests carry correctness proof, lanes carry speed.
3. dsv41 and the Qwen engine cannot coexist (RAM/VRAM): DS batteries need the 8081 lane free.
4. Falsified, do not re-run: CACHE_THREADS=4, DSPARK_MAX=12, L3-warm, draft-overlap,
   shadow-quant (pack already 1.75-2.31 bpw mixed), DS4_STORE_ZC=1 (measured slower +
   timing-dependent bits), naive DS4_AHEAD=1 re-arm (in-code measurement 4.5 vs 4.7 tok/s).

## The moves (ranked by value/cost)

### D1 — Single stream drain: fold sampling into forward() (cheap, hours)
Facts: `forward()` drains the whole 40-layer round at ds4_model.cu:9059, returns, does host
bookkeeping (`cache_tick`, `eg_stats`, dead `seed_drafter` walk), and only then does serve launch
`sample_tokens` + a second `cudaStreamSynchronize` (ds4_model.cu:9716-9721) on an already-empty
stream.
Do: launch the sampler stream-ordered at the end of forward (logits are device-resident), keep one
drain, move `cache_tick`/stats after the token readback.
Exit: one sync per token; solo lane +0.5-1.5% (pure critical-path latency removed per token).

### D2 — Overlap moe_down with the lane wait: FALSIFIED IN CODE 2026-10-30 (do not build)
The win_experts path (DS4_MOE_TOKEN=0) already runs the down rows before the wait for the CPU
lane, and the member comment at ds4_model.cu:1026-1028 records the measurement: "the same bits,
no faster." Moving a ~100 µs GEMV into a multi-ms lane wait saves nothing — the wait itself is
the term to shrink (D3/D5 attack it directly). CLOSED without a battery.

### D3 — Store miss sourcing: page-cache/tmpfs A/B (cheap test, potentially the biggest DS lever)
Facts: store demand reads prefer O_DIRECT (ds4_pool.hpp:207-210: `pread_direct` first, buffered
`pread_par` only as fallback). O_DIRECT bypasses the page cache — on a box whose cache holds the
whole 131 GB pack (184 GB RAM; 155 GB buff/cache observed), misses physically hit the SSD while
the same bytes sit in RAM. The in-code DS4_AHEAD comment names "the SSD is the narrow part"
(ds4_model.cu:5047-5049). The Qwen shadow store proved this exact pattern: tmpfs placement of the
miss source was +20.8% conc (F-367 A/B). The DS "no shadow" verdict was about QUANT (a lighter
artifact is pointless), never about PLACEMENT — this test is new.
Do: env arm to prefer the buffered path (or tmpfs-copy the expert shard region when RAM allows —
131 GB does NOT fit beside the Qwen shadow; DS-alone boots have ~90-100 GB usable, so a partial
tier or the buffered path is the realistic arm); warm the cache (`cat` the shards once) in the
launcher; battery.
Exit: miss latency from ~100-500 µs (SSD) toward ~10-20 µs (RAM); honest est. +5-20% all lanes
if the miss term is as large as the accounting correction implies. Measure first via D9's split
telemetry before sizing the battery.

### D4 — Kill the O_DIRECT read-back memmove (bug-adjacent waste)
Facts: every store read lands O_DIRECT into an aligned scratch, then a full-length
`std::memmove(j.keep, j.dst + lead, j.len)` (ds4_pool.hpp:215) undoes sub-4K file alignment —
~2× DRAM traffic per store-read byte, on the bus already at 93-96% of the ~50 GB/s ceiling.
Do: make each matrix's slot offset file-aligned by construction (slot base = layer offset mod
4096), so `keep` is a no-op; delete the memmove.
Exit: halves store-read DRAM cost; est. +1-3% on miss-heavy rounds, more under conc.

### D5 — Lookahead, refined (NOT the naive re-arm)
Facts: DS4_AHEAD is force-disabled when the store is enabled (ds4_model.cu:5047-5049); the in-code
measurement says naive guesses cost 4.5 vs 4.7 tok/s (62% guess accuracy; SSD contention). The
guess machinery (`route_ahead`/`ahead_ask` → `store->prefetch`, ds4_model.cu:6841-6844,
9500-9508) exists but is dead in the standing config.
Do: ONLY after D3 (if the SSD stops being the narrow part, the 4.5/4.7 verdict must be
re-measured, not assumed). Then: demand-order prefetch (guesses queued BEHIND demand reads,
canceled on demand collision), store-tier only, depth 1. Fresh battery; treat the in-code number
as a falsification of the naive arm, not of the mechanism.
Exit: L+1 store latency hidden behind L compute; unknown until D3 lands — could be the largest
single DS move or still negative. Honest status: blocked on D3's verdict.

### D6 — Round front-edge janitorial (free µs, one commit)
- Engram: cache the two `Tensor*` per Engram layer at load; kill the per-token string building +
  fresh `unordered_map` + extra `eg_cache_mu` takes (ds4_model.cu:5834-5844; layers 1 and 14 every
  round).
- Fold `eg_stats()`'s per-round mutex pair into the request-end stat line (ds4_model.cu:9055).
- Hoist a `has_drafter` bool; skip `tap()`'s `std::find` + `seed_drafter` wholesale when
  `g.n_mtp == 0` (ds4_model.cu:8998-9000, 9065 — dspark never loads in serve).
- Batch the per-token `printf("T %d\n")` + `fflush` (ds4_main.cpp:883-886): token lines into a
  small buffer, flush per N tokens or on a timer, protocol unchanged.
Exit: a few µs + ~10 allocs off every round's front edge; no lane estimate (too small to bank),
taken as hygiene that compounds with D1/D2.

### D7 — Lane answer-path allocations (inside the GPU spin)
Facts: per layer per token with CPU-lane picks, `serve_lane` heap-allocates `picks`
(ds4_model.cu:6904), `StoreHold` builds three vectors + an O(k²) `std::find` dedup
(ds4_model.cu:6174-6183), `cpu_experts_mx/iq` build `std::function` closures; store waits poll
`io_->wait_any(2000)` (ds4_store.hpp:189) at 2 ms condvar granularity.
Do: per-ring-slot preallocated scratch (kRing is fixed), span entry lists, spin-then-wait with
`cpu_relax` for the first few hundred µs.
Exit: ~1-3 µs × ~40 layers off the lane answer latency (which IS the GPU wait); est. +1-2%.

### D8 — Store shrink path audit (correctness, not speed)
`store_watch` shrinks 0.7 GB chunks below 4 GB free RAM every 16 windows (ds4_model.cu:6606-6615);
the swap path's stale-`zc_tab` fix (comment 6660-6662) has no evident equivalent on the shrink
path in ds4_store.hpp. zc is OFF standing, so exposure is latent, but a shrink+re-arm boot could
dangle. Do: targeted audit + a shrink-under-load test before any zc experiment, ever.

### D9 — Round profiler + truthful telemetry (prerequisite for D2/D3/D5 batteries)
- Separate the drain: record an event after the last MoE combine and another after the head GEMV
  so `head_ms` and lane-wait time split in the async path (kills the DS-2 artifact permanently;
  today the "maya ds4 decode" decomposition prints "routing waits 0.00" while lane waits hide in
  head).
- This is the T55 queue item (arm the ds4 round profiler) — promote it to first, since every DS
  move above is sized by what it reveals.
Exit: per-round decomposition that attributes lane wait, store wait, GPU compute, and head
separately on the STANDING config; no perf change.

### D10 — CUDA graphs (deferred deliberately)
Zero graphs in dsv41 (no cudaGraph matches under src/dsv41/). Blockers are enumerable host-baked
scalars, not dynamic shapes: mailbox seq counters as kernel args (ds4_model.cu:9496-9497, 9534,
8915-8916), per-step positions (`fk.pos` 9294, `fi.pos` 9307-9308, AttnRows 9314-9326,
`lim.n[t]` 9354-9360), `++win_seq` 9128, `++head_seq` 8955-8956, sampler seed/counter 9708-9715,
Engram host IO 9035-9039. ~1200 launches/round at T=1.
Status: DO NOT START until D1/D2/D3 land — in a sync-bound round the graph win (enqueue tail +
launch gaps) is small next to lane latency, and capture work is wasted if the round shape changes
under D2. Re-price after the D-moves bank.

## Order of attack

D9 (see the truth) → D1 (one drain) → D4 (memmove kill) → D3 (miss sourcing A/B; sizes D5) →
D2 (moe_down overlap) → D6/D7 (janitorial, ride any boot) → D5 (refined lookahead, gated on D3) →
D8 (audit, alongside any zc question) → D10 (graphs, re-priced last).

## Rollback registry (fill as moves land)

| Move | Env / binary rollback | Gate result | Date |
|---|---|---|---|
| D1 | sampler-launch env off / pre-d1 | | |
| D2 | overlap env off / pre-d2 | | |
| D3 | buffered/direct env back / pre-d3 | | |
| D4 | alignment env off / pre-d4 | | |
| D5 | DS4_AHEAD stays unset (status quo) | | |
| D6 | janitorial env off / pre-d6 | | |
| D7 | scratch env off / pre-d7 | | |
| D8 | audit only — no engine change | | |
| D9 | telemetry additive — no gate | | |
| D10 | graphs env off / pre-d10 | | |
