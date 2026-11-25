# The first request after a boot decodes at 60-75 % of its settled speed: mechanism and priced levers (F-467 follow-up, 2026-11-25)

Operator, 18:40: "go study how we can make that better ... There should be no 'no'."  Written while the operator
tested on the live standing server; code reads and the F-467 boots only, no boot of my own.

## 1. The measurement (campaign/cold-cost-probe.py, five boots, `campaign/results/ctx-f467-cold-*`)

| | first request | again, same shape |
|---|---|---|
| solo, ~3.8k-token prompt, 256 tokens | 141 / 143 / 168 / 175 t/s | 190 / 230 / 238 / 231 t/s |
| tier hit on its decode lookups | 90 % (thinking phase), 78-80 % (answer phase) | 95-96 % |
| share of routed experts read over PCIe | 2.2-2.3 %, then 5.6-6.8 % | 0.7-0.8 % |
| decode window | 8.5 ms | 5.3 ms |

The request held at the door until the warm-in finished decoded at 146.2 t/s (`ctx-f467-hold`): the same
first-request cost.  It lasts about one request: 0.4-0.7 s on a long generation, about 0.1 s on a tool call.

## 2. The mechanism (static reading, confirmed by the terms above)

- One writer feeds the tier's routing counts: the DECODE dispatch, `src/core/expert_source.cpp:5272`
  (`d.usage[layer * n_expert + id] += 1` for every routed id of a verify window).  The prompt read, which computes
  the routing of 1.2k-97k tokens right before decoding, writes nothing to it.
- The standing runs `--adapt-decay 0.85` and the default `--adapt-min-gain 1.5`: an expert is admitted when its
  decayed count beats the expert it replaces by 1.5x.  With the warm-in's counts still large, a hot new expert
  needs on the order of 40-50 tokens to displace a stale one (`generate.cpp:11138` applies the decay a round).
- A miss is a 2 MB expert blob over PCIe on the window's path: 5 % of 495 lookups a token is about 25 blobs, 50 MB,
  1.0 ms a token at 49.6 GB/s: the measured +3 ms on a 5.3 ms window.
- The adaptation round already takes a second input beside the counts: a predicted-usage vector (`pred`,
  `STRATA_PREDICT_CREDIT`, the lookahead predictor; `generate.cpp:11000-11006`, `adapt_scan(usage, pred, heat)`).
- The warm-in ends with a six-way burst, so the tier is fitted to the burst when live traffic arrives.  Its own
  line says so in four boots: the short right after the corpus ran at 104-123 tokens/s over the whole request,
  the same short after the burst at 98-106.

## 3. Levers, cheapest first

| # | lever | where | price | what it should move | test |
|---|---|---|---|---|---|
| 1 | end the warm-in on the corpus (burst before it) | `serve/server.py` `_replay` order | one reorder | the tier's end state from burst-fitted to prose-fitted: +4-15 % on the first prose request (the settled-short evidence) | hold-test boot + cold-cost boot: first request 146 -> ? |
| 2 | faster tier response: `--adapt-decay` 0.85 -> 0.7, `--adapt-min-gain` 1.5 -> 1.2 | config | more swaps in steady state; F-465 found the concurrent lane swap-rate-bound | first request +10-30 %; concurrent lane must hold | config ladder, 4 boots a knob, same hour |
| 3 | boot on the operator's own routing: `--expert-profile` = a snapshot of what the engine learned in a real session | config, no code | none at run time; the snapshot must be taken before a battery re-pins the learned bin (F-271) | the first request meets a tier fitted to real traffic: the "second request" hit rate, 95 %+ | the operator restarts on the snapshot and sends the same first request |
| 4 | prefill-informed admission: count the prompt's routing into `pred` and run one adaptation round before the first decode window | engine: the prefill's per-layer top-k ids -> a device histogram (layers x experts int32, ~100 KB) -> one copy -> `pred += w x histogram`; env-gated, default off | one swap round before decode: ~50-300 blobs, 0.1-0.6 GB, 2-12 ms | removes the deficit at its cause, for every request whose content shifts (first request, thinking -> answer, after a tool result), not only the first after a boot | candidate binary + the gauntlet (bit-identical fingerprints: the tier changes which expert computes a row, never the result) |

F-464's negative on the learned profile (concurrent -3.5 to -6.8 %) was a profile of synthetic boots, scored on
the probe's own traffic; lever 3 is a profile of real traffic, scored on real traffic.

## 4. Order

1 and 2 are config and server changes: two boots each, run when the box is free.  3 needs thirty minutes of the
operator's real use first (the engine saves `data/expert-profile-learned.bin` every 30 minutes between requests).
4 is the structural fix and the only one that reaches requests that are not the first after a boot; it is priced
at 100x its cost (0.4-0.7 s removed for 2-12 ms spent) and needs the engine's binary ladder.

## 5. Open

- How well a prompt's routing forecasts its answer's routing: not measured.  A cheap read: per-request tier hit
  on the answer phase against the hit the prompt's own histogram would have given (a diagnostic counter, no swap).
- Whether a round of swaps before the first window disturbs concurrent sessions sharing the tier (bounded by
  `--adapt-swaps`; measure on the 6-conc lane).
- The admission record ("1 generated") still hides batch-lane generations from every reader of the engine's line.
