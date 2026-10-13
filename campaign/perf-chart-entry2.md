
## ENTRY 2 — 2026-10-12 17:17–17:45 — the instrumented boot (degradation bounded, concurrency recorded)

Window: boot **17:17:30**, the standing recipe with `STRATA_SPLIT_TIMING=1` + `STRATA_DECODE_TIMING=1`
(temporary diagnostic). Lane: quiet machine, GPU 32 GB, no throttle.
Sources: `campaign/results/load-instrumented-1717.txt`, `campaign/results/watch-standing-1715.txt`,
`strata-iq3_s.log` (boot L44702).

| # | metric | this session | previous best / benchmark | verdict |
|---|---|---|---|---|
| M1 | single-stream, 200-token requests | **136–167 t/s** (n=218 across 5.8 min + mixed bursts) | 145.8 (this session's A/B leg); 280.7 all-time (4.83 drafts/win) | at class |
| M1b | single-stream, 2000 tok | 145.8 (17:04 leg) | as above | at class |
| M2 | 6-conc aggregate | **176.5 / 176.8 t/s** | 105 on the degraded 16:33 battery; 196–205 campaign best | **+68% vs the degraded battery** |
| M2b | pair aggregate | 141.4 t/s | 108.8 (this session's leg 2) | up |
| M3 | acceptance | 90.2% (A/B leg 2) | 90.8% (leg 1) | at class |
| M6 | window cost, per window | **wait 9.100 + pool+plan 1.224 + staging 0.201 + commit 0.085 = 10.6 ms** over 25,544 windows | model 9.7 + 2.47·(T−1) | at floor |

### What entry 2 establishes

1. **The degradation did not reproduce**, and is now bounded on three sides: 210 sustained sequential
   requests (5.7 min) — no decay (it improved: 145 → 167 t/s); 2-conc + 2×6-conc bursts — no decay
   (single-stream *rose* to 152–163 t/s after); a 54,819-token fresh prompt — no decay.
2. **Every internal candidate is exonerated with numbers** (adaptive exchanges, conversation evictions,
   KV index lookups, commit pressure, GPU throttle, CPU load, host staging/commit path). The counters
   that looked causal were correlating with time: the *good* boot ended with twice the exchanges and
   three times the evictions of the degraded one.
3. **The instrument is now permanent in the shared launcher**, so the next slow episode names its own
   stage in one log read: `wait for the GPU` up ⇒ the GPU is being time-shared; `pool + plan` or
   `host staging` up ⇒ the host path.
4. **Concurrency is not a regression** on this build: 176 t/s aggregate at 6-conc, measured after the
   6-conc paths had been though, and it does not leave the single-stream lane degraded.

### Chart direction

M2 up (176 vs 105), M1 at class, no metric down. The 2x bar is still unmet: 145.8 / 232. The named
lever remains **draft yield** (tokens per window) — the window itself is already at its floor
(10.6 ms measured, 9.7 + 2.47·(T−1) predicted) and now instrumented to prove it.
