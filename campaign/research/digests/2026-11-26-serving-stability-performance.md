# Serving stability & performance — web digest (2026-11-26, operator directive: "research the web of these things and other low-level inference engine mechanics")

Scope: what the field says about the exact failure classes and levers this lane hit in the last day —
repetition-loop handling, expert-cache/offload mechanics at high hit rates, speculative-decoding
stability, KV reuse on long agent conversations, request-shape economics, and operational stability.
Every item is screened against this tree's banked facts (laws 12, 15). Companion digests: the
overnight F-469 sweep (`2026-11-26-mtp-specdec.md`, `-engine-performance.md`, `-kv-cache.md`,
`-sampling-stability.md`, `-loops-precision.md`) and `2026-11-26-gpu-cache-hit-99.md`.

## Lane 1 — repetition handling in serving (the F-468 lesson)

- **No serving stack we found ends a reply from output-side repetition detection.** Industry practice
  puts repetition control at sampling time (repetition/frequency/presence penalties, per-token), not a
  post-hoc word-n-gram coverage kill. Our F-468 answer-phase guard was a local invention, and F-478
  reverted it: word-level 12-gram coverage cannot separate CSS boilerplate (0.26–0.364) from
  degeneration (0.263) — the ranges overlap at ANY threshold.
- **Vendor guidance is phase-specific:** Qwen's model cards recommend `presence_penalty` 1.5 for
  *thinking-mode* generation and `presence_penalty` 0 for *precise coding* (the qwen35moe card in our
  own tree says exactly this, `serve/server.py:8817`). Our F-470 put 1.5 on the thinking card, which
  merges into every thinking-card request — including the operator's code answers at temperature 0.3.
  Scoping candidate: apply the card's penalties to the thinking phase only, answer phase penalty-free.
- **Scoped guard design space (F-478 open item):** (a) thinking-phase coverage retained, answer phase
  exempt; (b) prose-gated coverage (only spans without code-fence/language markers counted); (c)
  per-request opt-out. (a) is the minimal-risk scope; (b) is the honest metric fix.

## Lane 2 — expert cache & offload at high hit rates (our 95–96% window band)

- **Bevy** (arXiv:2611.04794): *"prefetching can be harmful when there is an expert cache in GPU
  memory — as the cache includes more necessary experts, expert prediction becomes less precise.
  Proper request scheduling beats prefetching."* Runahead scheduling of concurrent requests (a ready
  request runs ahead and reveals the experts it needs next) beats prefetch prediction: up to 3.46×
  llama.cpp decode throughput. **Direct challenge to the FS_SLOTS arm in the hit-99 queue: at our
  95–96% hit rate the predictor's precision ceiling is low; the runahead/scheduling shape (cohorts,
  `STRATA_COHORTS`) is where the tree already has machinery (law 5).**
- **Adaptive miss assignment** (arXiv:2610.23006, Lv et al.): route GPU-missed experts to CPU *or*
  PCIe-transfer by token load, balancing both paths to finish together — **+39.5% decode, +51.9%
  prefill over the strongest baseline**. This is the F-125/126/127 lane exactly: our CPU-expert misses
  are 2.3–3.8% of routed entries (2.9–4.1 ms/window) and the untested structural lever is overlap.
- **IsoMoE** (arXiv:2611.27604): hybrid CPU-GPU MoE with **bitwise-identical outputs across expert
  placement** (a GPU kernel that replays the CPU kernel's arithmetic) and a **dual micro-batch
  pipeline that overlaps CPU and GPU compute without changing outputs** — 1.26× avg / 1.54× max
  decode. Two prizes for us: the overlap structure for the miss path, and placement-invariance as a
  canary-grade stability contract (M3b's exact-match law, extended).
- **SAEM** (arXiv:2608.21614): consecutive reasoning stages in chain-of-thought exhibit coherent
  expert-activation patterns; stage-aware caching + in-situ CPU execution = 1.33–1.54× on reasoning
  workloads. Our thinking-phase streams are exactly this shape — stage-aware residency windows.
- **Edge0** (arXiv:2609.18063): a per-layer prerouter predicts the next layer's routing one token
  ahead and the prediction IS the routing (staged set = routed set, nothing dropped) — the design our
  RouterLookahead lacks (`warms()` false on the resident tier is a structural absence, per the
  gpu-cache-hit-99 digest).
- Also screened: CCAMoE (2611.28550, cache-conditioned speculative drafting, 2.33×), OLED-MoE
  (confidence-guided inter-iteration retention), SPICE (2608.21240, low-rank surrogates for
  low-confidence misses, 3.12×), HiNa-MoE (2610.05123, CPU AMX/AVX expert kernels up to 2.09× e2e),
  AdaOffload (2611.10219), FreeToken (2608.16157, edge-native elastic CPU-GPU), CAEE (2606.29982,
  cost-aware expert pruning). Community signal: llama.cpp's LRU expert-cache work claims up to 2.2× on
  one GPU (video-level lead; the PR is the artifact to read before pricing anything).

## Lane 3 — speculative decoding stability (our 58–89% acceptance band)

- **Draft calibration** (arXiv:2611.33111): community draft models are systematically
  **underconfident** relative to the target; post-hoc **temperature scaling of the draft distribution
  improves acceptance length ~10% at near-zero cost**. Cheapest arm on this digest: a draft-temp
  calibration curve (we swept temp DOWN and falsified sharpening, F-471 75.9→63.9; this paper's
  mechanism says the gain direction is calibration, not sharpness — the arm is a curve, not a point).
- **HSM / HSGD** (arXiv:2611.34092): training-only methods for MTP drafters — Hidden State Mixing
  (best at later draft positions), Hidden State Gradient Detach (best at early positions); up to
  **+52.2% mean acceptance length over co-trained MTP**. Feeds the queued FastMTP retrain cycle
  design directly (train_cycle.py owner).
- **AsymSpec** (arXiv:2611.34010): acceptance rises monotonically within a generation (the draft
  aligns in-context) — another reason depth ladders (not fixed depth) pay.
- **Sightline** (arXiv:2611.23807): on *code generation* (SVG-as-code), block drafters' "misses are
  rare and clumped" and accuracy persists far deeper into a block than on natural language — longer
  verification windows pay once candidates are wide. Maps to our F-228 depth stair +
  `STRATA_SPEC_DEPTH`: a code-shape depth arm (deeper windows while the stream is in a code block).
- **FACT** (arXiv:2611.23463): *"changing only host-thread settings can nearly eliminate speedup
  without changing acceptance"* — execution conditions are first-class; our aux-cpus/threading
  placement is in that class. Also: confidence-guided candidate allocation beat static allocation
  1.46×→2.02× at fixed budget.
- Also screened: NarrowSpec (2611.23835, draft hidden width ½–¼ of target, +6–44% at batch 1),
  Pando (2611.32269, noise-coupled trees, +8.4% acceptance), quantized draft KV (2611.20565, K4V4,
  −66% draft cache at −1.36% relative acceptance), DFly2 (2611.34776, drafter shares target KV+index).

## Lane 4 — KV reuse on long agent conversations (NEW finding, priced)

- **Observed on the operator's stream today:** after a 47,618-token generation, the next request's
  50,156-token prompt reused only **2,534 tokens** (5%) and re-read 47,622 in 7,549 ms (6,308 tok/s).
  The next request again matched exactly 2,534. The conversation cache retains only the fixed
  prelude; every agent turn re-prefills the whole history. At ~6.3k tok/s prefill this is ~7.5 s of
  pure re-read per turn — the single biggest per-request latency term on long streams.
- **Callo** (arXiv:2611.18459): agent sessions with context edits invalidate cached KV; they serve
  cache-miss requests by *speculating on the invalidated KV during re-prefill* (1.05–1.19× all-request
  on SWE-agent traces). Precedent for making re-prefill useful; the first question for us is simpler —
  why our append-only turns do not hit the cache at all (commit path? slot eviction? length-finish
  handling? client re-serialization?). Code read owed: the conversation-cache commit logic and slot
  policy in `serve/server.py` + the engine's checkpoint accounting.
- Context: vLLM automatic prefix caching and SGLang RadixAttention are the production comparables
  for prefix-tree KV reuse; our conversation slots are the local analog.

## Lane 5 — request-shape economics (the 130 t/s meter)

- Community signal ("Tokens Per Second is a scam for AI agents", video-level lead): agent clients
  read tokens/wall-time, so short capped requests show fixed-cost amortization, not engine speed.
  Our arithmetic already owns this: 1-token requests divide the per-request fixed cost (~8–14 ms +
  prompt read) into 92–137 tok/s; `max_tokens 200` with thinking capped to 100 (answer_room +
  SMALL_CAP rules, server.py:275–276, 8803) lands ~130 effective. Engine decode was 190–225 tok/s
  throughout. Remediation is metering honesty (per-phase tok/s in the done-line already exists) plus
  client-side `max_tokens` guidance — not an engine change.

## Lane 6 — operational stability (already law, restated with sources)

- Boot churn: 33 restarts in 24 h each paid door-hold (≤180 s) + warm-in (5.5 min with corpus).
  F-478 addendum: the live-traffic gate becomes code (battery-lib), not discipline.
- Thermal: F-474 soak (board 3 °C under alarm after ~8 h synthetic load, M1 −10–20%, recovered).
  Standing response: same-hour comparison law + a board-temp read wired into lane probes.
- THP: F-472 adopted (+7.5% M1, KV shadow site only); complement organ stays double-locked.

## Ranked queue (cheapest falsifier first; law 4/15)

1. **KV-reuse collapse code read** (free; potentially multi-seconds per request): why exactly 2,534
   tokens match — commit path, slot eviction, length-finish, client serialization.
2. **Draft-temp calibration curve** (one env arm; paper-priced ~+10% acceptance): the F-471 sweep
   falsified *sharpening*, not *calibration*; run the curve the paper describes.
3. **Bevy cross-check on the FS_SLOTS queue** (free; re-prices the hit-99 arm): at 95–96% hit,
   predictor precision is structurally low — confirm against our ForesightSwap code before arming.
4. **Code-shape spec depth** (env arm on a code stream; Sightline + F-228 stair): deeper windows
   while inside a code block.
5. **Presence scoping off the answer phase** (server.py edit beside operator hunks; needs code-quality
   verification, not just acceptance): vendor card precedent in-tree at :8817.
6. **Adaptive CPU-miss assignment / overlap** (engineering; F-125 arithmetic + 2610.23006 +39.5%
   and IsoMoE's dual micro-batch pipeline): the structural lever the refutation boundary left open.
7. **Scoped loop guard (thinking-only)** (server.py edit; F-478 open item).
8. FastMTP retrain with HSM/HSGD findings folded into the cycle design (GPU-exclusive, operator go).
