### F-200 — THE TIER-0 BATTERY SWEEP: THREE ARMS ADOPTED (device rescue, async launch, resident-margin 0.75) + the six-brief research haul (2026-10-15 ~03:45, goal round 18 execution)

**THE SETUP.** 11 boots, one battery, ~50 min, zero collapsed runs (the run-2 stall did NOT
appear).  Control → 9 arms → bracket control; the window FELL −4.9/−8.1/−8.6% end-to-end, so
every arm is scored against a time-interpolated control (both controls kept in results/).
Deep-trough window (ctl M1 136.4 vs 164.8 two hours prior; M3b 62-65 band).

**THE RESULTS (M1/pair/6-conc vs interpolated control):**

| arm | delta | verdict |
|---|---|---|
| STRATA_TYPICAL_DEVICE=1 | **+9.0/+6.7/+6.4** | **ADOPTED** (parity-proven device rescue port; prefill proxy −8.7% too) |
| STRATA_VERIFY_ASYNC_LAUNCH=1 | **+4.9/+7.9/+8.9** | **ADOPTED** (v0.1.44 HostGraphLaunch; attacks the 69% GPU-reach line) |
| ROUTE_RESIDENT 0.75 (from 0.5) | **+14.9/+8.5/+12.2** | **ADOPTED after 2× MMLU gate PASS** (92.86/97.14, mean 95.00 — the good band; routing-changing arm, watch note stands) |
| ATTN_LANECELL=1 | +3.6/+5.4/+1.9 | below M1 bar — shelved (phase-table check pending) |
| HC_SPLIT=0 | **−15.4/−16.6/−11.8** | REJECTED hard (the 41-block layout IS the bad one, as F-194 predicted) |
| HC_SPLIT=1 / GR_V3=1 / L2 72 / MMQ_KQ=1 | all ≤±2.5% | neutral today — MMQ needs a prefill-lane test, not these metrics |

**THE CONFIRMATION BOOT** (td+val stacked, fresh boot, 3 probes): **M1 157.1 / pair 158.6 /
6-conc 196.5, M3b 80.6** — vs the end control (129.7/124.5/180.3): **+21.1% M1 / +27.4% pair /
+9.0% conc**, and M3b recovered from the window's 62-65 band to 80.6.  The full stack (td+val+
rr75) is now STANDING (launcher updated; server rebooted healthy); the next fresh window's first
battery re-baselines it.  Three adoptions in one night = the campaign's biggest single round.

**THE SIX RESEARCH BRIEFS** (2-at-a-time agents per the stream-limit law; full texts in
campaign/research/F-200-*.md — 66 findings total).  The items that CHANGE THE ROADMAP:

*Attention (4.40 ms #1 line):* SpecSA row-group index sharing (up to 6.86× kernel — our exact
shape); TMA Gather4 CONFIRMED working on sm_120a (replaces serial dependent reads); one-line
cp.async.bulk.prefetch.L2 of the next block (1.84-2.15× kernel, L2 hit 0.06→43.7%); SageAttention
int8-MMA (our KV is already int8 — stop the scalar dequant+shuffle path); the 5090 FA recipe's
mma fragment layout = zero cross-lane traffic; **GDN BV-tile retune worth ~10% e2e on our exact
model family (SGLang PR #17094)**.

*MoE (4.90 ms doorbell tail):* llama.cpp graph-safe expert-slot indirection (+84% class — the
closest implementation of our T1.1); KTransformers one-layer-late CPU-expert deferral (+1.45×
class); next-layer expert prediction at 90-99% recall (two variants); HOBBIT T₂-skip of
unimportant misses; LLRU eviction formula (~5 lines); VERIFIED: SM120 MoE kernels must be
Marlin-class (TMA-WS broken, 99KiB smem); our tail is overhead-dominated (sub-ms compute).

*Spec decoding:* **draft temperature down-scaling (draft at T'≈0.4-0.5 while target serves 0.7 —
training-free, lossless, +3-5pts acceptance plausible) = the cheapest untested lever in the
campaign**; Block Verification (ICLR'25, +3-8% lossless, drop-in); Nightjar length bandits
(+14.8%); depth 7-8 + early stop (verify rows ~free on a launch-bound engine); suffix-automaton
draft override (TRT ships it).  The exact-lossless space = block verification + traversal
verification + multi-draft-without-replacement.

*Graphs (the 9.5 µs node spacing):* Hazy Research's 1.3 µs graph floor → our spacing is ~7×
floor = mostly recoverable; **fence/doorbell audit is the suspected dominant tax
(__threadfence_system 0.5-2 µs/node → possibly 9-12 ms/window)**; vertical fusion 1,800→700-900
nodes = 8-10 ms; device-side tail launch (constant latency, no host); conditional nodes GREEN
FIELD CONFIRMED (nobody ships them); SET on RTX 5090: host overhead worse on Blackwell.

*Prefill:* layered prefill SOURCED (arXiv:2510.08055: chunked prefill inflates MoE expert loads
+39%; up to 70% TTFT); Green Contexts P/D (+60% on a 5070 Ti; 5090 power-wall caveat);
libsmctrl TPC masks; SPF reordering (free when prompts queue); **VERIFY OUR GDN PREFILL IS
CHUNKWISE (sglang-jax #1416: per-token recurrent scan was their dominant long-prefill cost)**.

*Quant:* QTIP trellis 3.1-3.3bpw should beat IQ3_S quality at −7-10% size; IQ3_KS in mainline
(cheapest A/B); Kitty 2-bit KV (8× KV density, tested on Qwen3, page-centric = our shape);
QuantMoE-Bench allocation recipe (rank by outlier+frequency, first blocks high-bit);
**exact two-phase lm-head: NOBODY SHIPS IT — the Cauchy-Schwarz-bound construction is provably
exact (521MB → 30-70MB/window)**; 2:4 sparsity for the hc line (~1.3-2×).

**ROADMAP v2 (queue after tonight's adoptions):**
R25 draft-temp down-scale (env/1-line, lossless) → R26 fence audit (cheap, suspected 9-12 ms)
→ R27 GDN BV-tile retune (constant change, ~10% e2e class) → R28 block verification (lossless
drop-in) → T1.1 partial plan (llama.cpp slot-indirection template) → T1.2 attention row-fold
(SpecSA template + L2 prefetch + int8-MMA) → KTransformers deferral → layered prefill →
conditional nodes / device tail launch → exact two-phase head → Kitty KV / QTIP (rebuild tier).
Full set: campaign/ROADMAP-30-VS-STOCK.md + research/F-200-*.md.

Sources: campaign/results/f200-*, mmlu-mmlu-f200-adopt{,-b}.json.
