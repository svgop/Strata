
### F-181 — THE FULL-CONTEXT QUESTION ANSWERED: native 262k measured, 1M YaRN built and measured — ladder on record, 131k stays standing (2026-10-14 ~03:30, goal round 11)

**THE QUESTION (operator).** We run `--max-context 131072`; the model "supports 1M with yarn" — test
full context and the performance difference.

**THE GROUND TRUTH FIRST.** The GGUF declares `context_length = 262144` (rope base 1e7) and carries
NO rope-scaling keys — 1M is not in the file; it is Qwen's served recipe applied at load time. The
engine (v0.1.44 merged) has full support: `--rope-scaling yarn --rope-scale 4` (llama.cpp names;
boot confirms "yarn, factor 4, freq_scale 0.25, mscale 1.138629 against a trained context of
262144"). Test legs: 262k native (`custom-iq3s-262k.json`) and 1M YaRN ×4 (`custom-iq3s-1m.json`),
both from the standing battery; new instrument `campaign/depth-lane.py` (one request: deterministic
synthetic filler to target depth + fixed decode; server timings parsed; M6-class).

**THE LADDER** (same binary, back-to-back boots; "cold" = fresh conversation, "warm" = the
deterministic filler let the conversation cache serve a prefix — noted per row):

| config | depth | prefill t/s | decode t/s | accept |
|---|---|---|---|---|
| (short-ctx class, same binary) | ~1-30k | ~5.7-6.5k | ~195 (M1) | ~79-81% |
| 262k native | 157k (cold) | 6534 | 116.7* | 62.4%* |
| 262k native | 244k (warm, cache_n 131k) | 5711 fresh | **173.9** | **79.4%** |
| 1M yarn ×4 | 128k (cold) | 6291 | 126.3 | 73.7% |
| 1M yarn ×4 | **937k** (warm, cache_n 113k) | **4343** | **98.7** | 60.9% |

\* the 157k cold row decoded only 266 tokens (early stop) — short sample, lower-confidence row.

**READING.** (1) Depth costs decode roughly linearly in the sparse-index overhead band we can see:
−11% at 244k (best case: warm cache, 79% acceptance), −36% at 937k (98.7 t/s — still ~7x the public
ceiling for this class). (2) Prefill: ~6.3-6.5k t/s in the 128-157k band, −33% at 937k (KV/idx
streaming grows with depth; YaRN itself ≈ −4-8% at matched depth, 6291 vs 6534 cold-cold). (3) Draft
acceptance degrades with depth on synthetic filler (79% → 61% at 937k) — how much is filler-content
vs rope-extended positions is NOT separated; a real-corpus long-context probe would tell. (4) RAM
headroom is a non-issue: 129 GB used of 184 at 937k with the 49 GB conversation cache absorbing the
streamed KV. (5) **QUALITY: YaRN ×4 is clean at short prompts** — MMLU 94.29 ± 2.77 under the 1M
config (mid-band vs tonight's 92.86-95.71 standing spread; mscale applies globally so this gate was
necessary). Long-context REASONING quality (needle-style) at ≥262k is UNMEASURED — flagged, not
claimed.

**VERDICT.** The 1M and 262k configs are BUILT, TESTED, and on the shelf (launchers
`launch-custom-iq3s-262k.sh` / `-1m.sh`) for any deep-context workload: at 937k the lane still
serves 4343 t/s prefill / 98.7 t/s decode. The STANDING lane stays 131k — the campaign's bars and
the anchor are measured there, and full-context costs 11-36% decode by depth. Sources:
`campaign/results/depth-{262k-d130k,262k-d240k,1m-d127k,1m-d965k}.txt`, `mmlu-1m-yarn-ml.json`.
