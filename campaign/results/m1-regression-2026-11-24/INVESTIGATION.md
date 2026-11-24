# M1 "regression" investigation — 2026-11-24 17:05-18:20

Trigger: the re-pathed standing read M1 210.5 / 208.1 (two boots, three measured legs each) against the
same day's 14:08 bench of 236.9 (`11-bench/BENCH-NUMBERS.md`). The operator asked for the factors and
for optimizations that preserve quality. Every claim below is a measurement with its control; the two
hypotheses I carried into the session (steering raises draftability; start latency after conc) were both
falsified by the instruments and are recorded as such.

## 1. Nothing in today's work costs M1 — the config ladder (law 17, one window)

`campaign/config-abab.py` (new: the config counterpart of the binary ladder; standing launcher's own env
arms, two configs, A/B alternating, learned profile PINNED to one file before every boot). A = the standing
minus the phase args (legacy path), B = the standing as adopted. Same binary, dose 0.3, 4 boots x 3 legs
(leg 1 discarded), 17:35-17:50. `campaign/results/m1-phase-2026-11-24/`.

| lane | A unscoped | B phase-scoped | A vs B | band |
|---|---|---|---|---|
| M1 | 217.8 | **220.2** | -1.1% | +-4% |
| 6-conc | 259.9 | 270.5 | -3.9% | +-4% |
| M3b accept | 76.1% | 76.6% | -0.6% | |

Phase scoping is M1-neutral. Acceptance is identical either way: the persona never bought speed through
draftability (falsified: "steering makes reasoning more draftable"). B = 220.2 is the F-408 bank (220.6).
Exonerated by direct control: the learned profile (pinned to the 13:57 file = the 14:08 bench's state;
same read), the DATA re-path (slots are touched at park/restore only; B ran on it and read the bank),
thermals (SM 2917-2925 MHz, 46-48 C across every burst and after; `gpu.json` in each anatomy dir).
The 14:08 window read +7% over the evening windows on the same recipe: the box's documented +-8%
across-hours drift (law 17). A high window, not a different standing.

## 2. What the M1 number actually contains — the anatomy instrument

`campaign/probe-m1-anatomy.py` (new): the lane's own M1 request (`probe-lane.py` REAL, 2000 tokens),
streamed, TTFT separated from decode, S S / burst / S S / burst / S S in one boot, GPU clocks sampled.
The lane's M1 is tokens / full wall of ONE request that always follows other prompts - it is, by
construction, the first request of its routing signature in every leg.

| run | S1 | S2 | S3 (after burst) | S4 | S5 (after burst) | S6 | after vs quiet |
|---|---|---|---|---|---|---|---|
| sampled 0.7, conc burst | 189 | 232 | **196** | 227 | **194** | 231 | -11.4% |
| sampled 0.7, single-stream burst of equal duration | 197 | 205 | **202** | 236 | **203** | 224 | -6.0% |
| greedy 0, conc burst (identical text every request) | 186.5 | 186.4 | **165.8** | 182.4 | 185.6 | 202.6 | -7.3% |
| greedy 0, STRATA_ADAPTIVE_SPEC=0 | 172 | 187 | 181 | 198 | **171** | 188 | -5.5% |
| **greedy 0, burst of the SAME prompt (control)** | 189 | 195 | **193.5** | 188 | **188.8** | 180 | **+1.6%** |

- Thermals: out (clocks/temperature flat). Concurrency/shape: out (single-stream burst reproduces it).
- Sampled content: out (greedy reproduces it with identical text; S1 = S2 to 0.1 t/s).
- TTFT: out (0.07-0.11 s on the slow requests; the fast ones often WAIT longer, 0.33 s).
- W1 depth controller: out (reproduced with STRATA_ADAPTIVE_SPEC=0; its counters are cumulative with one
  update site on the serial path - a burst cannot move them; W1 off also costs ~7% outright).
- Signature change: **the control** - a burst of the same prompt leaves the next request at full speed.

## 3. Mechanism: the adaptive expert tier's reaction lag on a routing-signature change

Per-request engine lines, greedy run (identical text, so window counts are stable at 644-691):

| request | decode ms | ms/window | expert-cache hit | experts read over PCIe |
|---|---|---|---|---|
| fast | 7251-7400 | 11.0-11.3 | 97.5-96.0% | 1045-3788 |
| slow (first after a signature change) | 7661-8308 | 11.8-12.6 | 94.7-94.8% | **2-3** |

A slow request fetches essentially no experts to the GPU: its misses are computed on the CPU-expert path
(the flush breakdown carries CPU experts 3.66 vs 2.65-2.97 ms/window and GPU-reach wait 1.26 vs 0.42).
The tier swaps experts in from DECAYED usage, a capped number per adapt round; a burst of another
signature decays the returning signature's usage to ~0, so the first returning request starts from zero
and re-earns residency a few swaps per round for most of its length. The second request finds it resident.
(An earlier reading of mine - "constant ms/window, more windows" - was taken from a two-line sample of a
different run and is retracted: with identical text the window count is stable and the cost per window moves.)

Correction of record: this retracts nothing in the adoptions' capability evidence, but it exposes a gap -
neither engine adoption (two-table `0bd36187e`, phase gate `ba13d1acb`) recorded an M1 probe at adoption
time (law 3 requires it for token-stream-affecting changes). The config ladder above closes that gap
retroactively for both: neutral.

## 4. Levers, priced (law 15) - none shipped in this round

- **L1 - prefix-memoized residency (structural, law 6; the conversation-cache idea applied to residency).**
  The cache already recognises a returning prefix ("prompt 1176 tokens = 1171 reused"). Park each
  conversation's last expert-usage vector with it and seed the tier's `usage` from it on restore: the first
  adapt scan of a returning signature swaps the right experts in at once. Needle: the first request's
  re-residency transient, measured 1.0-1.4 s per 2000-token request (-10..-15%); the lane's M1 request is
  always first-of-signature, so the lane reads the transient every leg: expected **+8-10% M1** (toward
  the 228-232 warm rate), conc unchanged, routing and numerics untouched (a format extension to the slot
  file through its owning module, old slots seed zero). Precondition check: prefill cannot supply the
  signal here - the prefix is reused, so its routing never recomputes; the cache must carry the knowledge.
- **L2 - faster tier reaction** (`--adapt-swaps` up, `--adapt-decay` down): a config ladder away, but it
  buys re-residency with more PCIe traffic mid-request and risks the B6 thrash the pins exist to prevent.
- **L3 - prompt-routing seeds residency**: fails for cached prefixes (no prefill compute); subsumed by L1.
- **L4 - the instrument's honesty**: the M1 lane includes one signature-change transient by construction.
  Not changed here (the bank's contract); named so its swings are read correctly.

Validation owed for L1 when it ships: slot-format round-trip parity, the anatomy control (post-change
request at warm rate), the config ladder on M1/conc/M3b, and the MMLU gate + M3b canary (CPU and GPU
expert paths may differ in rounding; exact-match verification decides).
