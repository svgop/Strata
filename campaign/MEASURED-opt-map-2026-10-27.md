---
type: contract
id: measured-optimization-map-2026-10-27
status: evidence
---

# Measured optimization map — live self-telemetry pass, 2026-10-27

Source: the engine serving this chat (PID 251743, iq3_s, spec 6, batch 6, ctx 131072, kv int8,
kv-resident 32768, resident-experts, mtp, prefill auto:32768). Read-only pass; no code touched.
Evidence: `/metrics` (JSON + text), `/tmp/m3.jsonl` (450 samples @0.32 s), `/tmp/gap_before.txt`
/`gap_after.txt` (cumulative `strata:token_gap_seconds` delta), `custom-iq3_s.log` per-request
blocks, WSTAT window flushes, and three live probes run against the serving engine:
`/tmp/probe_stream.py` (per-token SSE arrival traces), `/tmp/probe_cost.py` (window wall-time
cost model at 1/2/4/6 lanes × 2 context sizes), `/tmp/probe_ctx.py` (9k→72k context curve),
`/tmp/probe_deep.py` (10 Hz metrics + 5 Hz nvidia-smi clock/power/PCIe sampler),
`/tmp/probe_interleave.py` (6 decode lanes + injected cold reads, per-lane ITL segments),
the 80k→108k attention-knee / 2k–32k prefill-knee series (`/tmp/probe_knee2.py`),
`/tmp/probe6.py` (idle-ramp/transport), `/tmp/probe7.py` (clock/power isolation), and
full-session mining of the 11,373 per-request blocks in `custom-iq3_s.log`.

## Measured stack (per-turn anatomy, solo lane, 60–76k ctx)

A turn = tool-call round-trip. Time from request arrival to last token:

    admit ~0 ms  →  fixed prompt setup 317–934 ms (see M9 — regime-dependent)
    →  decode at ~209 tok/s (median), gapped by ~19.3 ms window-boundary gaps

Per-token arrivals during decode: 67.3% of gaps <0.5 ms (spec accepts inside a window, ~3
tokens per burst), 32.7% at 10–50 ms (mean 19.28 ms — one per window boundary). WSTAT for the
current session: p50 23.129 / p99 29.596 / max 36.054 ms, 363 windows / 3,486 rows (9.6 rows
per window). So window wall time is the dominant term of the boundary gap; host-side inter-
window dead time is small and cannot be separated from these two numbers alone — the lever is
shrinking the window itself, not just overlapping host setup.

## M1 — Fixed per-turn prompt overhead: 317–460 ms/turn (biggest felt-latency term)

Measured: 122 requests session totals prompt_ms 84,844 (avg 695; min 109, max 484); the last
7-request sampler window: 3,225 ms / 7 req = 461 ms each for reads of only 26–172 tokens.
Conversation cache keeps warm TTFT 0.08–0.79 s, but the reuse-scan + park/restore + join/promote
sequence costs ~300–460 ms per tool-call round-trip before the first token. At ~15 turns/turn
this is ~5–7 s per chat exchange — larger than the decode of a short answer.
Where: W8 (server lane: admission overlap, PROMOTE-without-STOP at generate.cpp:16477–16516,
park/restore costs server.py:1944/1976, generate.cpp:9656+ / 11678–11700) + W11 (post-idle
tier-swap storm, join 2.46 ms, promote 12–17 ms, park 24–52 ms, restore 12.4–15.6 ms).
How: keep the session warm across idle gaps — elastic demote fired 12 times this session
(last_demote 41.2 ms); each demote forces the next turn to pay restore + join.
What: W11 spare-pool warmth above demote watermark + W8 in-engine PROMOTE; then measure the
per-request fixed prompt_ms term directly (it is visible in the `requests` block) as the
acceptance metric.
Expected: TTFT −30–50% per tool-call round-trip; no t/s change (throughput-neutral, felt-speed
move).

## M2 — Window wall time: p50 23.1 ms, ~4 tokens per window

Measured: 2,624 tokens in 12,373 ms decode (212.1 tok/s) against 1,861 boundary gaps →
~4 tokens per window at ~19.3 ms wall. Window p50 23.1 ms matches the F-218 heavy-class cost
map (24.31 ms). Acceptance 89.3% (1,870/2,095) is healthy; the window is long, not wasteful.
Where: W1 (adaptive speculation — trim window when the tier says marginal depth is dead),
W2 (ungated draft chain), W13 (launch-count fusion: Mix/Combine + LFUSE + router/plan/bell),
W10 (QSA decode attention retune), W9 (S26 expert dequant on sm_120, env-only).
How: every ms off the window wall time is off every boundary gap — 4 tokens/window × 19.3 ms
means −10% window ≈ −10% per-lane decode t/s directly.
What: W9 first (env-only, +6–10% bit-identical), then W1/W2/W13.
Expected: W1–W13 are the roadmap's core; this measurement confirms the window wall time is
where the per-token time goes (23.1 ms wall / 4 tokens ≈ 5.8 ms/token effective).

## M3 — Decode rate slide with context: 210 → 190 tok/s from 60k → 76k

Measured: per-request decode 187–231 tok/s across ctx 60,825 → 76,028; KV VRAM hit decays
99.49% → 98.98% over the same range; expert hit 96–97%.
Where: W10 (QSA decode attention retune — +3–12% at 8K+, more at 64K), W6 decode split kernel
(folded into W10), W11 (spare-pool warmth slows the KV-hit decay).
How: the ~10% slide over 16k ctx is the attention+KV-miss term growing with context; W10
attacks it directly at the kernel level.
Expected: recovers the 190 end of the band at 76k; deeper at 128k.
Correction (M15/M16/M23): the window wall time itself is context-insensitive below ~72k —
solo depth-8 windows at ctx 60, 18k, and 60k sit at p50 22.46 / 22.12 / 23.1 ms. But the
session-wide log (M23) shows acceptance RISES with context (0.773 at 0–9k → 0.88–0.89 at
60–90k), so the slide cannot be acceptance drift. The slide is the window wall-time context
term (M20: +4.7–7.6 ms above ~80k) plus the chronic clock deficit (M17). W10's share of
this 10% is real again at long context; W1 owns the depth-8 row waste, not the slide.

## M4 — MoE fetch wait: 7–22 ms in 78–195 fetches per turn

Measured: log lines "decode waited X ms in Y fetches" — 7–22 ms/78–195 fetches on some turns,
0.0 ms/0 fetches on others (resident experts mostly hit; misses are tail experts).
Where: W4 (hide the CPU-expert rendezvous, conc +10–17%, solo/pair +3–8%), W9 (dequant
kernels), W13 (router/plan/bell fusion — fewer fetches per window).
How: 22 ms of wait on a ~12 s turn is ~0.2% — small for solo, but the same waits are the
contention source in pair/conc lanes (W4's lane).
Expected: solo ~0–2%, pair/conc +10–17% per W4.

## M5 — Decode-rate jitter: 126–294 tok/s across 0.32 s intervals

Measured: interval rates p25 190 / median 209 / p75 234 / max 294 / min 126; the tok_s gauge
ramps 132 → 236 over the first second of each turn (moving average, not a real stall) and
drops to 12–56 right at window boundaries (sampling artifact).
Where: W11 (jitter package), W1 (adaptive speculation — the tier swap itself is the variance
source: adapt tier-swap cap 96/round, join 2.46 ms).
How: the 126–294 spread is acceptance-depth variance inside a turn — W1's controller is the
right owner; the post-idle swap storm is W11's.
Expected: smoothness (p99 ITL), not mean t/s.

## M6 — Prefill: regime-dependent rate (cold 68k at 6,512 tok/s → 10.5 s TTFT; small reads at 1,207–2,236 tok/s)

Measured: marginal prefill 6,032 tok/s; cold 68k read 6,512 tok/s; power spike 515 W during
prefill; 34–46 °C — latency-bound, not power/thermal-bound. The context-curve probe (M16)
fills the rate-vs-read-size curve: full reads of 9k/18k/27k/36k tokens at 4,804 / 6,432 /
6,766 / 6,047 tok/s; post-reuse reads of 12k–39k at 5,342–6,467 tok/s. So the launch-bound
regime is only below ~2k tokens (M9's 1,207–2,236 tok/s); above ~9k tokens the rate is
already ~5–6.8k tok/s.
Where: W7 (MMQ device grouping + GDN pipelining + cuBLASLt autotune, +10–25%), W14 (staging
pool double buffer + MMQ host-path kill, +10–25%).
Expected: prefill +10–25% → cold 68k TTFT 10.5 s → ~8 s; warm turns barely affected.

## M7 — GPU: util 92–100%, power 200–350 W decode, idle 88 W

Measured: decode is latency-bound at 200–350 W on a 5090 — headroom exists; the kernel is
waiting, not saturating.
Where: confirms W1/W2/W13/W10 direction (launch-count + latency reduction), rules out any
power/thermal move.

## M8 — Concurrency lane curve (measured live, 1/2/4/6 lanes — first direct measurement)

Probed the engine with staggered concurrent requests (small prompts, 128 max_tokens):

| lanes | aggregate tok/s | per-lane tok/s | window p50 | rows/window |
|---|---|---|---|---|
| 1 (chat) | 209 | 209 | 23.1 | 9.6 |
| 2 | 164 | 90.5 / 73.9 | 17.1 | 3.5 |
| 4 | 279 | 62–79 | 19.7 | 6.9 |
| 6 | 341.5 | 50–68 | 23.8 | 9.0 |

The GPU absorbs batch rows almost for free: window wall time stays 17–24 ms while rows/window
goes 3.5 → 9.0. Aggregate scaling 1→6 lanes = 1.63×; per-lane ITL at 6 lanes is 15–20 ms.
Batch-lane TTFT 0.03–0.62 s.
The batch lane is NOT spec-off: `--batch-mtp` runs 6 slot drafters (boot log line), and
rows/window ≈ 4 slots × (1+D) with D=1 (generate.cpp:5375 `STRATA_BATCH_DRAFT_DEPTH`, default
1, clamped ≤ min(spec−1,3)). Batch drafts are counted as verifier rows, NOT in the request's
`drafts_offered` field (the 5,462 "drafts accepted 0 of 0" log lines are a telemetry gap, not
missing speculation). Inferred batch acceptance from rows vs outputs: ~87% at 4 lanes.
Where: W3 (device acceptance on batch), W4 (rendezvous), W8 (admission overlap).
New lever: STRATA_BATCH_DRAFT_DEPTH=2/3 — env-only, batch_mtp already live, the campaign's
`batch-draft-depth-notes.md` has the ramp test; never measured on this engine.
Expected: at D=2 the batch lane carries 2 proposals per slot per window — if acceptance holds
near the solo 89%, the 6-lane aggregate should move from 341.5 toward ~450+.

## M9 — prompt_ms mechanism (corrects M1's "6,032 tok/s marginal" estimate)

Log lines give the actual decomposition: `prompt 60053 = 58925 reused + 1128 read in 934 ms`;
`prompt 63163 = 61220 reused + 1943 read in 869 ms`; cold 62-token read = 689 ms; warm-reuse
read = 21–36 ms. So prompt_ms is NOT one marginal rate — it is regime-dependent:
- cold first request: ~690 ms fixed setup (slot bring-up + restore + join — elastic demote
  fired 21.6–30.6 ms after the probes; restore itself is cheap: 58,925 tokens in 58.5 ms);
- warm partial reuse: ~20–40 ms for the reuse scan at small reads;
- large turn with ~1k+ new tokens: ~870–934 ms (reuse scan + read at 1,207–2,236 tok/s —
  small reads are launch-bound, 4–5× slower than the 6,512 tok/s cold-68k bulk rate).
Where: W11 (warmth: cold→warm is the 690 ms→30 ms jump), W8 (PROMOTE), W14 (MMQ host-path
kill), W15 candidate below (small-read prefill fast path).
Expected: the 690 ms cold setup is the single biggest felt-latency number on the map;
W11's spare-pool warmth + demote-watermark is its direct fix.

## M10 — Full latency quantiles (histograms, first pass measured)

TTFT: mean 0.777 s, p50 0.5, p90 2.5, p99 20 s (p99 = cold 68k prefills). ITL: mean 5.06 ms,
p50 5, p90 10, p99 20 ms. e2e: mean 5.56 s, p50 5, p90 20, p99 80 s. Gap: p50 0.5 ms, p90 20,
p99 30 ms. Admission: ~0 (p99 ≤ 0.5 ms). The p99 TTFT 20 s and p99 e2e 80 s tails are the
cold-prefill population — W7/W14 own them; the ITL p99 20 ms is the window-boundary tail —
W1/W2 own it.

## M11 — Decode gate + MoE fetch (log-level, first pass measured)

`decode waited 33.0 ms in 206 fetches` on the 60k turn; 8.1 ms in 13 fetches on small turns.
The fetch gate: "closed, closed in 58238 windows, 455 flips; budget 0 MB/window; read-ahead
used 100%" — the gate is closed for nearly every window, so the 33 ms is the tail that got
through. Expert hit 93.0% at 60k (vs 96–97% at smaller ctx); "cache room 1.30 of the experts";
36.47 GiB experts resident in RAM; file tier 20,453 blobs / 104.5 GB read since boot. KV
streaming: 98.09% of 7.84M block reads hit VRAM, 602 MiB from RAM.
Where: W4 (hide the rendezvous), W9 (dequant on sm_120), W11 (file-tier read-ahead after
drop_file_cache).

## M12 — Elastic / conversation cache / VRAM state (first pass measured)

Post-idle demotes cost 21.6–30.6 ms each (park 0.0, unmap 3.6–4.4, "the cache ate them back
and refilled from the profile"); spare_chunks 24. Conversation cache: 24/24 slots parked,
48 GiB budget, 18.2 GB held, 7 evictions, 69 parks / 6 restores, 141/154 requests reused,
skip-parking below the 200-token minimum. VRAM is nearly full: vram_free_mib 1244, arena
37,348 MiB, expert cache 20,521 MiB / 10,560 slots.
Where: W11 (spare-pool warmth above the demote watermark — the demote/re-restore loop is the
cold→warm penalty), W8 (PROMOTE). The 7 evictions at 24/24 parked slots mean the cache budget
is the binding constraint for long sessions — W11's budget sizing.

## M13 — History series smoothness (60-sample series, first pass measured)

gpu_util mean 64%, spread 92 pts; power mean 227 W, spread 324 W (91–416 W); temp 37–47 °C;
PCIe RX mean 6.0 GB/s, spread to 49 GB/s (prefill spikes); host cpu mean 20%. The power
spread is the idle→decode→prefill swing, not decode instability — confirms M5's jitter source
is the spec/acceptance cycle, not the power path.

## M14 — Streaming path: client gap ≈ server gap (the send path is not the hidden cost)

Probed the SSE path with per-token arrival timestamps at 1/2/4/6 lanes: client gap count
matches the server's `strata:token_gap_seconds` delta to −3…−18 (read chunking coalesces
bursts, never inflates them); client median ITL 8.07 / 12.27 / 17.30 / 21.78 ms at
1/2/4/6 lanes; slow-lane p50 17.09 / 13.04 / 18.40 / 22.93 ms; slow p99 729 / 165 / 218 /
329 ms. The per-token `data: ...` + flush loop (server.py:6935–6950) adds nothing
measurable on loopback — the felt ITL is the window wall time, not the transport. W8's
SSE-batching idea is throughput-neutral for latency; drop it from the latency argument.

## M15 — Window wall-time cost model: wall ≈ 8.4 + 1.57×rows ms, context-independent

Six clean window samples (rows/p50, from the cost-model probe): (3.1, 12.79), (3.3, 14.22),
(6.0, 16.93), (6.3, 19.59), (8.7, 22.12), (9.0, 22.46). Least-squares fit: wall = 8.4 +
1.57×rows ms, R² ≈ 0.99. The same rows at ctx 60 and 18k give the same p50 (22.46 vs
22.12), so context does not enter the window cost up to ~72k. (The 10k n=1 sample carries
a stale WSTAT flush — identical to the previous batch's window; flagged, harmless to the
fit.)
Row efficiency (tokens per window = tok_s × p50, divided by rows): solo spec-depth-8 lanes
run 30–41% efficient — 2.73 tokens at 9.0 rows at ctx 60, 3.61 at 8.7 rows at 18k; batch
D=1 lanes run 78–91% — 2.74/3.1, 5.32/6.0, 7.93/8.7. Consequence: the solo lane's depth-8
speculation wastes ~6 rows per window. Trimming depth to ~4 at these acceptances puts the
window at ~16.3 ms for the same tokens: 2.73/16.3 = 167 vs 121.4 tok/s, 3.61/16.3 = 221 vs
163.1 — +35–38% per-lane on solo lanes, bounded by the tokens a depth-4 chain still
accepts. That is W1's number, measured rather than inferred. W15's upside is smaller than
M8 guessed: the batch lane is already ~91% row-efficient at D=1 — D=2 adds rows the lane
can barely convert.

## M16 — Context curve: TTFT grows linearly with prompt_ms; cross-conversation reuse caps at the kv-resident window

Grew one conversation 9k → 72k tokens, 8 turns:
- TTFT = prompt_ms + ~70 ms: 1.96 → 6.84 s across the curve. The p99 TTFT population
  (M10) is this linear term — W7/W14 own it.
- Cross-conversation prefix reuse caps at exactly 32768 tokens = `--kv-resident`: turns
  5–8 each reused exactly 32768 and read the rest at 5,342–6,467 tok/s. The chat's own
  turns reuse 58,925+ because the conversation cache restores the full conversation —
  within-conversation reuse is uncapped, cross-conversation reuse is capped at the
  resident window.
- Full reads (no reuse) at 9k/18k/27k/36k: 4,804 / 6,432 / 6,766 / 6,047 tok/s (M6).
- Confound: probe turns co-tenanted with this chat's live decode (same engine), so the
  tok_s/acceptance columns of the probe are batch-lane numbers, not clean solo numbers;
  the window wall time and prompt_ms/prefill columns are structural and unaffected.
Where: W7/W14 (prefill rate), W16 (small-read fast path, scoped to reads <2k by M6/M16),
W11 (the 32768 cap — spare-pool warmth raises effective cross-conversation reuse).

--- Fifth pass: fluctuation sources, clock ceiling, attention knee ---

## M17 — GPU clock ceiling: SM pinned ~6% below max by an always-on throttle flag

`/tmp/probe_deep.py` (10 Hz metrics + 5 Hz nvidia-smi, 295 samples across idle / 6-lane
burst / cold 30k prefill / 6-lane burst):
- `clocks.sm` steady 2917–2962 MHz, max 3135 MHz — a constant ~6% t/s tax on every
  window, every lane, prefill and decode alike.
- `clocks_event_reasons.active` = 0x400 ("Reliability") Active at every sample;
  SW Power Capping counter 640,120 µs cumulative (power limit 575 W, decode draws
  200–350 W, prefill spikes to 475 W — so the capping is episodic, the flag is not).
- `gpu_pcie_rx_mb` is instantaneous MB/s, not cumulative: prefill pulls PCIe RX to
  ~31 GB/s at 475 W / 100% util — the expert-fetch traffic is a real prefill cost term.
- Boundary (corrected by the parallel session): a root lock service (`strata-gpu-lock.service`,
  SM_CLOCK=3135 max) already exists and runs while the engine is up — the deficit is NOT a
  missing unlock. The driver derates to ~2940 MHz under load with only "Reliability"
  (0x400) active at 43 °C / ~305 W — a Blackwell hardware V-F guardband at the top of the
  V-curve; no nvidia-smi setting overrides it, and the power limit is not the binding
  constraint. The ~6% tax is a hardware floor, not a software lever (see
  [[strata-gpu-lock-service]]).

## M18 — Prefill↔decode interleave: measured ITL spikes and TTFT inflation; cadence is OFF by default

`/tmp/probe_interleave.py`: 6 streaming decode lanes (256 tok each) + cold 4k read at
1.8 s + cold 16k read at 4.3 s:
- Lane p50 held ~30 ms during the reads, but per-lane ITL spiked to 0.1–0.8 s — the
  perturbation is spiky, not a uniform slowdown. (Model streams `reasoning_content` —
  parse both delta keys, handle `[DONE]` before json.loads.)
- Injected read TTFT ballooned: 4k read → 6.61 s, 16k read → 6.88 s — 3–5× the standalone
  M22 curve at the same sizes.
- Code already documents the mechanism (server.py:325–339): "a decode window that also
  reads a prompt chunk cannot replay the decode graph, so one 32k-token prompt read can
  add seconds to every decoding request's inter-token time", cites vLLM PR #44558
  (P50 ITL 274.9 → 32.1 ms). The fix ships as `STRATA_PREFILL_CADENCE=1` +
  `PREFILL_CADENCE_N` (default 2) — and is **off** on the running engine.
- Related already-on mechanism: #656 BYIELD — a long read yields at a chunk boundary to
  a much shorter waiting prompt (server.py:1851–1876).
Where: new W17 (boot-flag cadence).

## M19 — Fluctuation at large conversations: demote park scales, reuse-scan cost varies 2–413 ms

`custom-iq3_s.log` grep (1,406 demote lines, prompt-read decomposition lines):
- Elastic demote park cost scales with conversation size: worst observed demote 299.0 ms
  (park 214.6 ms) on large conversations vs the 21.6–30.6 ms M12 measured on small ones.
  W11's demote-loop cost is not a constant — it grows with exactly the workload that
  needs it most.
- The reuse scan's own cost grows with context and varies hard: one 1-token read at 80k
  ctx cost 413 ms (2.4 tok/s); sibling 1-token reads cost 2 / 101 / 191 ms — a 2–413 ms
  variance band on turns that read nothing. 26,808-ctx 45-token read: 433 ms.
  Corrects M9's "warm reuse 21–36 ms": that was the small-conversation regime.
Where: W11 (warmth kills the scan and the park), M1's fixed-overhead term is
size-dependent, not flat.

## M20 — Attention knee: windows pick up a context term above ~72k

Continuing one conversation 80k → 88k → 98k → 108k with decode-only window blocks
(128 generated tokens per turn):
- ctx 80–108k, 3.3–3.4 rows/window: observed p50 18.4 / 19.2 / 21.2 ms against the M15
  prediction 13.6–13.7 ms — a +4.7–7.6 ms context term (~35% of the window) that the
  row model does not carry. Below ~72k the M15 model is ctx-free; above ~80k it isn't.
- Window p99/max widen at 100k: 29.5–32.8 ms vs ~22–24 ms in the low-ctx band — the
  jitter M5 tracks has a context component above the knee.
- Reattributes part of M3's decode slide back to window wall time (M15's "slide =
  acceptance drift only" correction was measured at ≤72k; above ~80k both mechanisms bite).
  M23 settles it: acceptance rises with ctx across the whole session — the slide is the
  window wall-time term plus the clock deficit, not acceptance drift.

## M23 — Session-wide log mining: acceptance RISES with context; reuse-scan cost scales with ctx; rescue machinery live

11,373 per-request blocks in `custom-iq3_s.log` mined:
- Decode tok_s median: 223 at 0–9k → 200–207 at 20–90k → ~200 at 60–80k: the ~10% slide
  spans the whole session, and it is NOT acceptance drift — drafts accepted/offered
  rises 0.773 (0–9k) → 0.88–0.89 (60–90k). Long conversations are more predictable.
- Reuse-scan cost (prompt_ms for reads ≤200 tokens, n=9,556): median 73 ms at <9k ctx →
  269 ms at 80–89k → 313 ms at 90–99k; p90 546, max 663. Roughly linear in ctx —
  this is the scan cost every warm turn pays, and it doubles between 40k and 90k.
- Speculation config (live env, `/proc/251743/environ`): `STRATA_TYPICAL_ACCEPT=0.2` —
  Medusa-style typical acceptance (arXiv:2401.10774, generate.cpp:348–359) rescued
  172,799 of 325,139 first-rejections (53%) — half of the acceptance rate is
  probability-based rescue, sampling traffic only. `STRATA_BATCH_RESCUE=0` keeps the
  batch-site rescue off: armed, MMLU dropped 95.7 → 91.4–92.9 (F-30b) — the quality
  cost of the rescue is measured, in the code. `STRATA_SPEC_DRAFT_TEMP=0.85`,
  `STRATA_SPEC_COUPLED=1`, `STRATA_SPEC_PROB=1`, `STRATA_TYPICAL_DEVICE=1` live.
  Suffix drafts (`--suffix-draft 3`, n-gram drafter) active: 4,677 summary lines.
- BYIELD (#656): 3 events across the session, all "not taken" — the yield mechanism
  never actually fired in practice.
- Elastic promote is cheap: materialize 1.1–6.6 ms from the spare pool vs demote's
  21–299 ms — W11's cost is entirely on the park side.

## M24 — Clock/power isolation: prefill drives power to the cap; deep throttling is transient, not chronic

`/tmp/probe7.py` — 10 Hz nvidia-smi sampling, three phases:
- Idle: 2970 MHz, 91 W, no cap flag.
- Cold 64k prefill: 523–531 W (limit 575 W), SW Power Capping flag active in 53/75
  samples, clock median 2910 / min 2895 — capped but barely down.
- 6-lane decode: clock median 2917 / min 2902, power median 362 / max 527.
- Worst-case overlap (six concurrent cold prefills + 64k read, probe6): clock min
  1785 MHz — a transient cap spike; window p50 that burst was 30.3 ms.
So two distinct clock costs: the chronic ~6% deficit from the always-on 0x400
"Reliability" flag (M17, root-only), and episodic cap spikes during prefill storms
(M24). W17's cadence also flattens the power spike — it serializes prefill admission,
which is the exact condition that produced the 1785 MHz transient.

## M25 — No idle ramp, no hidden transport cost; the real post-idle tax is eviction persistence

`/tmp/probe6.py` warm small requests (1-token read, 80k ctx) after 0/30/60 s idle:
prompt_ms 205 / 205 / 208 ms, server TTFT 0.23 s identical, client-hidden term 0.03 s
(wall − server TTFT − decode). No idle ramp penalty — the elastic demote→materialize
cycle is cheap (M23). The earlier ~0.8 s "hidden" client gap in the knee probe was a
decode-accounting artifact (tok_s denominator); streaming traces show client first-token
== server TTFT exactly, confirming M14.
The real post-idle tax: eviction persistence — after the conversation cache evicted, the
first request read all 80,057 tokens (prompt_ms 11,290 ms, TTFT 11.31 s). Eviction cost
is paid in full on the first turn after it (M21), not amortized.

## M21 — Conversation-cache eviction at multi-100k tenancy; reuse falls back to whole 32768-token blocks

Same probe series: once 80k/98k/108k conversations coexisted, the cache evicted:
- Reuse landed on exactly 65536 and 98304 tokens — 2× and 3× `--kv-resident` 32768:
  the fallback path matches whole resident blocks only, so up to 32,768 tail tokens
  re-read at bulk rate (up to ~5 s/turn at 100k ctx).
- One conversation lost its blocks entirely: full cold read 89k → 20.5 s wall.
- Refines M16: the "32768 cap" is really block-granularity fallback once the conversation
  cache evicts; within-conversation reuse stays uncapped while the cache holds.
Where: W11 (spare-pool warmth), and a live cost of long chats — this conversation sits at
85–86k and pays it every turn if it evicts.

## M22 — Prefill size curve, clean: ~1.0 s fixed launch tax + ~6.8k tok/s marginal

Unique random-word cold reads (no reuse, no co-tenancy): 2,051 tok → 1,989 tok/s (1.03 s);
7,957 → 4,222 (1.89 s); 31,562 → 5,910 (5.34 s). Marginal rate 8k→32k = 6,820 tok/s;
fixed term ≈ 1.0–1.3 s regardless of size. TTFT 1.16 / 2.03 / 8.58 s. This is the
W16 target shape quantified (the fixed tax, not compute, dominates reads under ~8k)
and the W7/W14 marginal ceiling (6.8k tok/s, PCIe-capped per M17's 31 GB/s spike).

## Enacted vs boundary (what was not measurable in this pass)

- Enacted: 1/2/4/6-lane probes (M8), full histogram quantiles (M10), log-level gate/
  MoE/cache extraction (M11), history series (M13), conversation-cache/elastic state (M12),
  streaming SSE traces (M14), cost-model window fit (M15), 9k→72k context curve (M16),
  clock/power/PCIe deep sampler (M17), decode+injected-read interleave probe (M18),
  demote/reuse-scan log decomposition (M19), 80k→108k attention-knee series (M20/M21),
  clean 2k/8k/32k prefill-size curve (M22), session-wide log mining of 11,373 request
  blocks (M23), idle-ramp/transport probe (M25), clock/power isolation probe (M24).
- Boundary: GPU clock unlock — `nvidia-smi -lgc` needs root; the 6% deficit (M17) is an
  operator action, not an agent action.
- Boundary: PSTAT/BSTAT need boot-time env (STRATA_SERVE/BATCH_TELEMETRY) — BSTAT unset on
  the running engine; batch-window row-level telemetry needs the next boot.
- Boundary: cold boot / restart — the engine serves this chat; cannot restart.
- Not yet enacted: STRATA_BATCH_DRAFT_DEPTH=2/3 (env-only, batch_mtp live) — next boot;
  the D=2 ramp test in `batch-draft-depth-notes.md` is ready to run.
- Not yet measured: the 1 s slot-free pickup poll (server.py:2038–2052) — visible only in
  multi-client join timing; the 50 ms stagger probe didn't hit it.
- Superseded: the planned ~30k large-prompt prefill staging probe — the M16 context curve
  already covers full reads to 36k and post-reuse reads to 39k.
- Confound: probe turns co-tenanted with this chat's live decode (the engine serves this
  conversation); the probe's tok_s/acceptance columns are batch-lane numbers, not clean
  solo numbers. Structural columns (wall time, prompt_ms, prefill) are unaffected.

## Order of attack (measured, not assumed)

1. W9 (env-only, +6–10% all lanes) — cheapest window-wall-time cut.
2. W11 (warmth + jitter) — attacks the M9 cold-setup 690 ms and the M19/M23 fluctuation
   directly: demote parks scale to 299 ms, the reuse-scan median grows 73 → 313 ms from
   0–9k to 90–99k ctx, and eviction costs a full cold re-read on the first turn after
   (M25: 11.3 s on an 80k conversation). The biggest felt-latency numbers are size-scaled.
3. W1 + W2 (window wall time) — M2's 23.1 ms is the per-token time sink; M15's row-
   efficiency fit gives W1 a measured +35–38% on solo lanes (depth-8 windows run 30–41%
   row-efficient; trimming to ~4 rows keeps the tokens, cuts the window to ~16.3 ms).
   M20 adds the >80k context term: +4.7–7.6 ms/window the row model cannot remove.
4. W10 (context slide) — M3; M23 settles the attribution: acceptance rises with ctx
   (0.77 → 0.89), so the ~10% tok/s slide is the window wall-time context term (M20)
   plus the chronic clock deficit (M17). W10's kernel-level retune is the direct owner
   at long context.
5. W8 (PROMOTE, admission overlap) — completes M9's fixed-overhead kill; M14 removes the
   SSE-batching argument from its scope (the transport is not the cost).
6. W7/W14 (prefill) — M6 + M22 (marginal ceiling 6.8k tok/s, ~1.0 s fixed launch tax),
   the p99 TTFT 20 s tail (M10), and the M16 linear TTFT term.
7. W4, W13, W6, W3, W5, W12 as already sequenced.
8. Next boot, env-only, three flags at once: `STRATA_PREFILL_CADENCE=1` (W17, kills the
   M18 interleave spikes), `STRATA_BATCH_DRAFT_DEPTH=2/3` (W15 — worth one measurement,
   not a roadmap slot: M15 shows the batch lane already ~91% row-efficient at D=1), and
   the boot-time telemetry envs that unlock PSTAT/BSTAT row-level windows.

## New candidates beyond W1–W14

**W15 — batch-lane draft depth.** M8's measured scaling (1.63× aggregate at 6 lanes, window
wall time flat at 17–24 ms) shows the GPU absorbs rows for free; the batch lane is capped at
D=1 (one proposal per slot). Raising D to 2–3 is env-only (`STRATA_BATCH_DRAFT_DEPTH`,
batch_mtp live), bit-identical at D=1, and the campaign's ramp test exists. The telemetry gap
(batch drafts counted as verifier rows, not `drafts_offered`) should be fixed alongside so the
acceptance rate is directly readable per request.
Downgrade (M15): the batch lane already runs 78–91% row-efficient at D=1 — D=2 adds rows
the lane can barely convert, so the expected aggregate move is well below the 341.5 → 450+
M8 guessed. Run the ramp for the measurement; do not plan on the gain.

**W16 — small-read prefill fast path.** M9: reads of 1,128–1,943 tokens run at 1,207–2,236
tok/s — 4–5× below the 6,512 tok/s bulk rate — launch-bound, not compute-bound. A per-window
small-read path (skip the full prefill staging machinery below ~2k tokens) would cut the
agent-loop marginal term. Open only after W8/W11 land: if the warm per-turn prompt_ms stays
>200 ms, the evidence is already in the `requests` block.
Quantified by M22: the curve is ~1.0–1.3 s fixed launch tax + 6.8k tok/s marginal, so the
W16 prize is the fixed term — a 2k read costs 1.03 s where 0.3 s is compute.

**W17 — enable prefill cadence (boot flag).** `STRATA_PREFILL_CADENCE=1`
(`PREFILL_CADENCE_N`, default 2) is shipped, off, and documented in server.py:325–339;
M18 measures the exact disease it treats on this engine — 0.1–0.8 s per-lane ITL spikes
and 3–5× TTFT inflation (6.6–6.9 s) when cold 4k/16k reads interleave with six decoding
lanes. Zero code change; one boot. Watch the tradeoff the waiver already encodes: cadence
delays prefill admission while decode is busy, so measure TTFT p50/p99 alongside lane ITL
(vLLM #44558 reported both better, but that engine's chunking differs). Pair with #656
BYIELD, which already handles the short-prompt case.
